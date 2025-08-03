
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        # print(f'The obj to frame idx {self.obj_to_frame_idx}')
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        # print(f'Frame idx {frame_idx}, Video idx {video_idx}')
        flat_idx = video_idx * self.num_frames + frame_idx
        # print(f'Flat idx {flat_idx.shape}')
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    standard_idx = [i for i in range(1,4)]
    for video_idx, video in enumerate(batch):
        # print(f'The video index (frames), {video_idx}')
        orig_video_id = video.video_id
        orig_frame_size = video.size
        
        for t, frame in enumerate(video.frames):
            temp_obj_id = []
            temp_seg = []
            objects = frame.objects
            for obj in objects:
                # print(f'Objects: {obj}')
                orig_obj_id = obj.object_id
                temp_obj_id.append(orig_obj_id)

                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))
                temp_seg.append(obj.segment.to(torch.bool))
            # Rearrange in a started order
            sorted_indices = sorted(range(len(temp_obj_id)), key=lambda i: temp_obj_id[i])

            # Rearrange values and masks using the sorted indices
            sorted_values = [temp_obj_id[i] for i in sorted_indices]
            sorted_masks = [temp_seg[i] for i in sorted_indices]

            standard_idx = set(standard_idx)

            temp_id = set(sorted_values)

            not_common = list(standard_idx.symmetric_difference(temp_id))

            temp_mask =  torch.zeros([512, 512])
            for i in not_common:
                sorted_masks.insert(i, temp_mask)
            for i in sorted_masks:
                step_t_masks[t].append(i)


    # print(f'This is the size of moasks { len(step_t_masks[t])}')
            # print(f'the object id per frame{step_t_obj_to_frame_idx}')
    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    # Keep the masks dimensions consistent.
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )




def mask_to_points(mask):
      random.seed(42)
      mask = (mask == 255).to(torch.uint8)
      ys, xs = np.where(mask == 1)
      num_points = len(xs)
      yn , xn = np.where(mask == 0)
      n_points = len(xn)

      indices_n = random.sample(range(n_points), 5)

      unsampled_points = [(xn[i], yn[i]) for i in indices_n]

      if num_points == 0:
            sampled_points = [(-1, -1)] * 5  # No points at all
      elif num_points < 5:
            indices = random.sample(range(num_points), num_points)
            sampled_points = [(xs[i], ys[i]) for i in indices]
            # Pad with -1s
            sampled_points += [(-1, -1)] * (5 - len(sampled_points))
      else:
            indices = random.sample(range(num_points), 5)
            sampled_points = [(xs[i], ys[i]) for i in indices]

      results = sampled_points + unsampled_points
      
      return results

# class MRIDataset(Dataset):
#       def __init__(self, video_folder, label_folder, img_list=None):
#             self.video_folder = video_folder
#             self.label_folder = label_folder                          
#             self.imgs = []   
#             print(os.listdir(self.video_folder))                                           
#             # self.lbls = []   
#             for img in sorted(os.listdir(self.video_folder)):

#                 if img_list:
#                         # print(img)
#                         matcher = img.split('-stk')[0]
#                         # print(any(matcher in x for x in img_list))
#                         # print(matcher,img_list, img)
#                         if img.endswith('.npy') and any(matcher in x for x in img_list):
#                               print('Goes')
#                               self.imgs.append(img)
#                 else:
#                         if img.endswith('.npy'):
#                               self.imgs.append(img)                        
#             # for lbl in os.listdir(os.path.join(label_folder,'labels_UNN'))

      
#       def mask_bbox(self, label):

#             # Convert to torch tensor if needed
#             if not isinstance(label, torch.Tensor):
#                   label = torch.tensor(label)
            
#             tumor_per_class_per_slice = []
#             for i in range(1, 4): 
#                   per_slice_sum = torch.sum(label[i], dim=(0, 1))  
#                   tumor_per_class_per_slice.append(per_slice_sum)
            
#             # This is the area per tumour for each slice
#             tumor_per_class_per_slice = torch.stack(tumor_per_class_per_slice)  
#             # print(tumor_per_class_per_slice[1])
            
#             # Find slices where all 3 classes are present
#             class_presence = tumor_per_class_per_slice > 0
            
#             # returns all the indices where all 3 class exist
#             valid_slices = torch.where(class_presence.sum(dim=0) == 3)[0]

#             if len(valid_slices) == 0:
#                   # find the slices that contain 2 tumors and but what class should be prioritized in this manner the most
#                   print("No slice contains all 3 tumor classes.")
#                   two_class = torch.where(class_presence.sum(dim=0) == 2)[0]
#                   print(f'Print the 2 classes {two_class}')
#                   if len(two_class) == 0:
#                         one_class = torch.where(class_presence.sum(dim=0) == 1)[0]
#                         if len(one_class) == 0:
#                               return [torch.tensor([[0.0, 0.0, 0.0, 0.0]])] * 3, 0
#                         total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
#                         best_slice = one_class[torch.argmax(total_tumor_pixels[one_class])].item()
#                         print(f"Best slice with 1 tumors: {best_slice}")
#                   else:
#                         total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
#                         best_slice = two_class[torch.argmax(total_tumor_pixels[two_class])].item()
#                         print(f"Best slice with all 2 tumors: {best_slice}")
                  

#             else:
#                   total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
#                   # find the slices where the tumor area is the highests
#                   best_slice = valid_slices[torch.argmax(total_tumor_pixels[valid_slices])].item()

#             print(f"Best slice with all 3 tumors: {best_slice}")

#             bbox = []
#             points = []
#             for i in range(1, 4):
#                   # Get mask for that class at the best slice (H, W)
#                   mask = label[i, :, :, best_slice]
#                   if torch.all(mask == 0):
#                         box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
#                         point = [(-1, -1)] * 10
#                   else:
                       
#                         box = masks_to_boxes(mask.unsqueeze(0))
#                         point = mask_to_points(mask)
#                   bbox.append(box)
#                   points.append(point)

#             return bbox, points, best_slice
      

      
#       def __len__(self):
#             return len(self.imgs)

#       def __getitem__(self, idx):
#             image = self.imgs[idx]
#             print(f'Here is the errro{image}')

            
#             label_name = image.split('-')[0:4]
#             label_name = '-'.join(label_name)
#             # print(label_name)
#             label = [i for i in os.listdir(self.label_folder) if (label_name in i and i.endswith('.npy')) ][0]
#             label = np.load(os.path.join(self.label_folder, label))
#             bboxes, points, frames = self.mask_bbox(label)

#             # Labels 
#             label = torch.Tensor(label)[1:,:,:,:]
#             bboxes = np.array(bboxes)
#             print(points)
#             points = np.array(points)


#             return idx, self.imgs[idx], label_name, bboxes, points, frames, label
      
