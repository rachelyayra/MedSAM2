# Load NUMPY image and labels, and return Images, labels and bounding boxs
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
import cv2
import logging
import json
import tifffile as tiff
from PIL import Image
import random


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

class MRIDataset(Dataset):
      def __init__(self, video_folder, label_folder, img_list=None):
            self.video_folder = video_folder
            self.label_folder = label_folder                          
            self.imgs = []   
            # print(os.listdir(self.video_folder))                                           
            # self.lbls = []   
            for img in sorted(os.listdir(self.video_folder)):

                if img_list:
                        # print(img)
                        matcher = img.split('-stk')[0]
                        # print(any(matcher in x for x in img_list))
                        # print(matcher,img_list, img)
                        if img.endswith('.npy') and any(matcher in x for x in img_list):
                              # print('Goes')
                              self.imgs.append(img)
                else:
                        if img.endswith('.npy'):
                              self.imgs.append(img)                         
            # for lbl in os.listdir(os.path.join(label_folder,'labels_UNN'))

      
      def mask_bbox(self, label):

            # Convert to torch tensor if needed
            if not isinstance(label, torch.Tensor):
                  label = torch.tensor(label)
            
            tumor_per_class_per_slice = []
            for i in range(1, 4): 
                  per_slice_sum = torch.sum(label[i], dim=(0, 1))  
                  tumor_per_class_per_slice.append(per_slice_sum)
            
            # This is the area per tumour for each slice
            tumor_per_class_per_slice = torch.stack(tumor_per_class_per_slice)  
            # print(tumor_per_class_per_slice[1])
            
            # Find slices where all 3 classes are present
            class_presence = tumor_per_class_per_slice > 0
            
            # returns all the indices where all 3 class exist
            valid_slices = torch.where(class_presence.sum(dim=0) == 3)[0]

            if len(valid_slices) == 0:
                  # find the slices that contain 2 tumors and but what class should be prioritized in this manner the most
                  # print("No slice contains all 3 tumor classes.")
                  two_class = torch.where(class_presence.sum(dim=0) == 2)[0]
                  # print(f'Print the 2 classes {two_class}')
                  if len(two_class) == 0:
                        one_class = torch.where(class_presence.sum(dim=0) == 1)[0]
                        if len(one_class) == 0:
                              return [torch.tensor([[0.0, 0.0, 0.0, 0.0]])] * 3, 0
                        total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                        best_slice = one_class[torch.argmax(total_tumor_pixels[one_class])].item()
                        # print(f"Best slice with 1 tumors: {best_slice}")
                  else:
                        total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                        best_slice = two_class[torch.argmax(total_tumor_pixels[two_class])].item()
                        # print(f"Best slice with all 2 tumors: {best_slice}")
                  

            else:
                  total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                  # find the slices where the tumor area is the highests
                  best_slice = valid_slices[torch.argmax(total_tumor_pixels[valid_slices])].item()

            # print(f"Best slice with all 3 tumors: {best_slice}")

            bbox = []
            points = []
            for i in range(1, 4):
                  # Get mask for that class at the best slice (H, W)
                  mask = label[i, :, :, best_slice]
                  if torch.all(mask == 0):
                        box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
                        point = [(-1, -1)] * 10
                  else:
                       
                        box = masks_to_boxes(mask.unsqueeze(0))
                        point = mask_to_points(mask)
                  bbox.append(box)
                  points.append(point)

            return bbox, points, best_slice
      

      
      def __len__(self):
            return len(self.imgs)

      def __getitem__(self, idx):
            image = self.imgs[idx]

            
            label_name = image.split('-')[0:4]
            label_name = '-'.join(label_name)
            # print(label_name)
            label = [i for i in os.listdir(self.label_folder) if (label_name in i and i.endswith('.npy')) ][0]
            label = np.load(os.path.join(self.label_folder, label))
            bboxes, points, frames = self.mask_bbox(label)

            # Labels 
            label = torch.Tensor(label)
            bboxes = np.array(bboxes)
            # print(points)
            points = np.array(points)


            return idx, self.imgs[idx], label_name, bboxes, points, frames, label

class MRIDataset3D(Dataset):
      def __init__(self, video_folder, label_folder, img_list=None):
            self.video_folder = video_folder
            self.label_folder = label_folder                          
            self.imgs = []   
            # print(os.listdir(self.video_folder))                                           
            # self.lbls = []   
            for img in sorted(os.listdir(self.video_folder)):

                if img_list:
                        # print(img)
                        matcher = img.split('-stk')[0]
                        # print(any(matcher in x for x in img_list))
                        # print(matcher,img_list, img)
                        if img.endswith('.npy') and any(matcher in x for x in img_list):
                              # print('Goes')
                              self.imgs.append(img)
                else:
                        if img.endswith('.npy'):
                              self.imgs.append(img)                         
            # for lbl in os.listdir(os.path.join(label_folder,'labels_UNN'))

      
      def mask_bbox(self, label):

            # Convert to torch tensor if needed
            if not isinstance(label, torch.Tensor):
                  label = torch.tensor(label)
            
            tumor_per_class_per_slice = []
            for i in range(1, 4): 
                  per_slice_sum = torch.sum(label[i], dim=(0, 1))  
                  tumor_per_class_per_slice.append(per_slice_sum)
            
            # This is the area per tumour for each slice
            tumor_per_class_per_slice = torch.stack(tumor_per_class_per_slice)  
            # print(tumor_per_class_per_slice[1])
            
            # Find slices where all 3 classes are present
            class_presence = tumor_per_class_per_slice > 0
            
            # returns all the indices where all 3 class exist
            valid_slices = torch.where(class_presence.sum(dim=0) == 3)[0]

            if len(valid_slices) == 0:
                  # find the slices that contain 2 tumors and but what class should be prioritized in this manner the most
                  # print("No slice contains all 3 tumor classes.")
                  two_class = torch.where(class_presence.sum(dim=0) == 2)[0]
                  # print(f'Print the 2 classes {two_class}')
                  if len(two_class) == 0:
                        one_class = torch.where(class_presence.sum(dim=0) == 1)[0]
                        if len(one_class) == 0:
                              return [torch.tensor([[0.0, 0.0, 0.0, 0.0]])] * 3, 0
                        total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                        best_slice = one_class[torch.argmax(total_tumor_pixels[one_class])].item()
                        # print(f"Best slice with 1 tumors: {best_slice}")
                  else:
                        total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                        best_slice = two_class[torch.argmax(total_tumor_pixels[two_class])].item()
                        # print(f"Best slice with all 2 tumors: {best_slice}")
                  

            else:
                  total_tumor_pixels = tumor_per_class_per_slice.sum(dim=0)
                  # find the slices where the tumor area is the highests
                  best_slice = valid_slices[torch.argmax(total_tumor_pixels[valid_slices])].item()

            # print(f"Best slice with all 3 tumors: {best_slice}")

            bbox = []
            points = []
            for i in range(1, 4):
                  # Get mask for that class at the best slice (H, W)
                  mask = label[i, :, :, best_slice]
                  if torch.all(mask == 0):
                        box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
                        point = [(-1, -1)] * 10
                  else:
                       
                        box = masks_to_boxes(mask.unsqueeze(0))
                        point = mask_to_points(mask)
                  bbox.append(box)
                  points.append(point)

            return bbox, points, best_slice
      

      
      def __len__(self):
            return len(self.imgs)

      def __getitem__(self, idx):
            image = self.imgs[idx]

            
            label_name = image.split('-')[0:4]
            label_name = '-'.join(label_name)
            # print(label_name)
            label = [i for i in os.listdir(self.label_folder) if (label_name in i and i.endswith('.npy')) ][0]
            label = np.load(os.path.join(self.label_folder, label))
            bboxes, points, frames = self.mask_bbox(label)

            # Labels 
            label = torch.Tensor(label)
            bboxes = np.array(bboxes)
            # print(points)
            points = np.array(points)


            return idx, self.imgs[idx], label_name, bboxes, points, frames, label
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
#                               # print('Goes')
#                               self.imgs.append(img)
#                 else:
#                         if img.endswith('.npy'):
#                               self.imgs.append(img)                         
#             # for lbl in os.listdir(os.path.join(label_folder,'labels_UNN'))

      
#       def mask_bbox(self, label):

#             # Convert to torch tensor if needed
#             bbox = []
#             if not isinstance(label, torch.Tensor):
#                   label = torch.tensor(label[1:])
#             print(f'The original shape of label: {label.shape}')
#             label = label.sum(dim=0)
#             print(f'The shape of label: {label.unique()}')
            
#             # Find the best slice

#             label_counts = (label != 0).sum(dim=(0, 1))  
#             print(f'label counts: {label_counts}')
#             # Get index of the slice with the most label pixels
#             best_slice = 80
#             # bbox = []
#             points = []
#             mask = label[ :, :, best_slice]
#             if torch.all(mask == 0):
#                         box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
#                         point = 0
#             else:
#                         box = masks_to_boxes(mask.unsqueeze(0))
#                         point = 0
#             bbox.append(box)
#             points.append(point)

#             return bbox, points, best_slice
      

      
#       def __len__(self):
#             return len(self.imgs)

#       def __getitem__(self, idx):
#             image = self.imgs[idx]

            
#             label_name = image.split('-')[0:4]
#             label_name = '-'.join(label_name)
#             # print(label_name)
#             label = [i for i in os.listdir(self.label_folder) if (label_name in i and i.endswith('.npy')) ][0]
#             label = np.load(os.path.join(self.label_folder, label))
#             bboxes, points, frames = self.mask_bbox(label)

#             # Labels 
#             label = torch.Tensor(label)
#             bboxes = np.array(bboxes)
#             # print(points)
#             points = np.array(points)


#             return idx, self.imgs[idx], label_name, bboxes, points, frames, label
      
class ACDCDataset(Dataset):
      def __init__(self, video_folder, label_file):
            self.video_folder = video_folder
            self.label_file = label_file                        
            self.imgs = [i for i in os.listdir(self.video_folder)] 
            print(f'{self.imgs}')
            # Open and read the JSON file
            with open(self.label_file, 'r') as file:
                  data = json.load(file)
            
            self.label_dict = data
            print(self.label_dict.keys())

      def __len__(self):
            return len(self.imgs)

      def __getitem__(self, idx):
            # Select the annotation dict that corresponds to the image id
            largest = 0
            frames = 0
            image = self.imgs[idx]

            # print(image)
            # print(self.label_dict['annotations'][0].keys())
            vid_dict = [i for i in self.label_dict['annotations'] if image in i['file_name']]
            # print(len(vid_dict))

            # extract the length of the segments info, select the frame with the most number of object ids
            for i in range(len(vid_dict)):
                  num = len(vid_dict[i]['segments_info'])
                  # print(num)
                  if largest < num:
                        largest = num
                        frames = i

            # print(f'The thing {frames, largest}')

            # extract the object id at each time
            frame_dict = self.label_dict['annotations'][frames]['segments_info']
            label_name = self.label_dict['annotations'][frames]['file_name']
            # print(frame_dict)
            frame_obj =  [i['category_id'] for i in frame_dict ]
            bbox = [i['bbox'] for i in frame_dict ]
            
            return idx, self.imgs[idx], label_name ,bbox, frame_obj, 0 , frames,  0

class DRIVEDataset(Dataset):
      def __init__(self, image_path, seg_path):
            self.image_path = image_path
            self.seg_path = seg_path
            self.imgs = [img for img in sorted(os.listdir(image_path))]
            self.lbls = [lbl for lbl in sorted(os.listdir(seg_path))]

      def __len__(self):
            return len(self.imgs)

      def __getitem__(self, idx):
            image_dir = os.path.join(self.image_path, self.imgs[idx])
            image = tiff.imread(image_dir)
            
            print(image.shape)

            label_dir = os.path.join(self.seg_path, self.lbls[idx])
            label = Image.open(label_dir)
            label = np.array(label)
            label = torch.tensor(label)  

            bboxs = masks_to_boxes(label.unsqueeze(0))

            points = mask_to_points(label)

            bboxs = np.array(bboxs)
            points = np.array(points)

            return idx, image, self.lbls[idx], bboxs, points, [0], label

class STAREDataset(Dataset):
      def __init__(self, image_path, seg_path, img_list=None):
            self.image_path = image_path
            self.seg_path = seg_path
            # print(sorted(os.listdir(seg_path)))
            if img_list:
                  self.lbls = [lbl for lbl in sorted(os.listdir(seg_path)) if f'{lbl}.npy' in img_list]
                  print(self.lbls)
                  self.imgs = []
                  for lbl in self.lbls:
                        for img_file in sorted(os.listdir(image_path)):
                              if lbl[:-7] in img_file:  
                                    self.imgs.append(img_file)

            else:
                  self.imgs = [img for img in sorted(os.listdir(image_path))]
                  self.lbls = [lbl for lbl in sorted(os.listdir(seg_path))]
            print(self.imgs)
      def __len__(self):
            return len(self.imgs)

      def __getitem__(self, idx):
            image_dir = os.path.join(self.image_path, self.imgs[idx])
            image = Image.open(image_dir)
            image = np.array(image)

            label_dir = os.path.join(self.seg_path, self.lbls[idx])
            label = Image.open(label_dir)
            label = np.array(label)
            label = torch.tensor(label)  

            bboxs = masks_to_boxes(label.unsqueeze(0))

            points = mask_to_points(label)

            bboxs = np.array(bboxs)
            points = np.array(points)

            return idx, image, self.lbls[idx], bboxs, points, [0], label


