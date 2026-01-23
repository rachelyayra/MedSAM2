# Class Wrapper for SAM2
import sys
sys.path.append("/scratch/radjoe/Projects/Experiments/SAM_DIR/sam2") 

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor, build_sam2
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp


class SAM2Model:
    def __init__(self, model_cfg, checkpoint, model_type, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model_type = model_type
        if self.model_type == 'video':
            self.model = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
        else:
            self.model = build_sam2(model_cfg, checkpoint, device=device)
        


    def predict(self, input, bbox, frames, points):
        predictor = self.model
        if self.model_type == 'video':
            
            inference_state = predictor.init_state(video_path=input)
            predictor.reset_state(inference_state)
            # backwards and then forward
            if points == None: 
                for i in range(len(bbox)):
                    predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frames,
                            obj_id=i,
                            box = bbox[i]
                        )
                    
            else:
                for i in range(len(points)):
                    labels = np.ones((len(points[0]),), dtype=np.int32)
                    print(labels.shape, points[i].shape)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frames,
                        obj_id=i,
                        points = points[i],
                        labels = labels
                    )

            video_segments = {}  # video_segments contains the per-frame segmentation results
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                # print(out_frame_idx)
                
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse = True):
                # print(out_frame_idx)
                
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }


            return video_segments
        else: 
            if points == None:
                predictor = SAM2ImagePredictor(self.model)
                input = np.array(input)

                predictor.set_image(input)

                for i in range(len(bbox)):
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox[i],
                        multimask_output=False,
                    )
            else:
                predictor = SAM2ImagePredictor(self.model)
                input = np.array(input)

                predictor.set_image(input)

                input_point = np.array(points)
                print(f' this {input_point.shape}')
                input_label = np.ones((input_point.shape[0]//2,), dtype=np.int32)
                input_zeroes = np.zeros((input_point.shape[0]//2,), dtype=np.int32)
                labels = np.concatenate((input_label, input_zeroes))
                print(labels)

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=labels,
                )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
            return np.expand_dims( masks[0], axis=0)








    