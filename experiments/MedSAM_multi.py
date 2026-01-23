## Predict And Save Prediction in Directory

# Import Relevant Libraries
import os
import sys
sys.path.append('/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/')

from datasets.datasets import MRIDataset, DRIVEDataset, STAREDataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from models.MedSAM2_multi import MedSAM2Model

import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from copy import deepcopy

import argparse
import torch




def postprocess_save(prediction,label_shape, model_type):
    
    if model_type == 'video':
        total_mask = np.zeros((3, 192, 224, 160))
        print(f'label shape {(label_shape)}')
        for out_frame_idx in range(0, 160):
            for out_obj_id, out_mask in prediction[out_frame_idx].items():
                print(out_mask.shape)
                print(total_mask[:,:,out_obj_id].shape)
                total_mask[out_obj_id,:,:, out_frame_idx] += out_mask.squeeze(0)
        return total_mask
    else: 
        return prediction



def predict_and_save(model_cfg=None, checkpoint=None, video_path=None, dataloader=None,savepath = None, model_type=None, model=None, dataset_type=None):
    '''
    Dataloader
    Predict
    Save to directory

    '''
    model = MedSAM2Model(model_cfg=model_cfg, checkpoint=checkpoint, model_type=model_type, model=model)
    # print(dataloader)
    inference_time_list = []
    for idx, img_path, label_name, box, point, frames, label in dataloader:

        label_shape = label.shape
        saves =  img_path[0][:-8]
        point = point.squeeze(0)
        box = box.squeeze(0)
        print(f'Print number of boxes: {box}')

        if model_type == 'video':
            img_path = os.path.join(video_path, img_path[0])
            video_dirs = img_path
        else:
            video_dirs = img_path[0]

        print(f'This is the path to the box {box.shape}')
        
        print(frames)


        prediction_point = model.predict(video_dirs, box, frames[0], point)
        point = None
        prediction_box = model.predict(video_dirs, box, frames[0], point)
        

        os.makedirs(savepath, exist_ok=True)

        save_path_box = os.path.join(savepath,'boxes')
        save_path_point = os.path.join(savepath,'points')

        os.makedirs(save_path_box, exist_ok=True)
        os.makedirs(save_path_point, exist_ok=True)

        save_path_box = os.path.join(save_path_box,f'{label_name[0]}.npy')
        save_path_point = os.path.join(save_path_point,f'{label_name[0]}.npy')


        output_box = postprocess_save(prediction_box,label_shape, model_type)
        output_point = postprocess_save(prediction_point,label_shape, model_type)
        np.save(save_path_box, output_box)
        np.save(save_path_point, output_point) 
    

    df = pd.DataFrame(inference_time_list, columns=['Sample', 'Time'])
    df.to_csv(f'{save_path_box}/inference_times.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation predictions and save masks")
    
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label directory')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save predictions')
    parser.add_argument('--model_cfg', type=str, required=True, help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, help='Whether to choose the video predictor or image predictor')
    parser.add_argument('--dataset_type', type=str, default='MRI', choices=['MRI', 'DRIVE', 'STARE'], help='Type of dataset')
    

    return parser.parse_args()     
        
def main():
    print('Start here')
    args = parse_args()
    if args.dataset_type == 'MRI':
        dataset = MRIDataset(args.video_path, args.label_path)
    elif args.dataset_type == 'STARE':
        dataset = STAREDataset(args.video_path, args.label_path)
    else:
        dataset = DRIVEDataset(args.video_path, args.label_path)

    dataloader = DataLoader(dataset, shuffle=False)
    print('Here it is')
    predict_and_save(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        video_path=args.video_path,
        dataloader=dataloader,
        savepath=args.save_path,
        model_type= args.model_type
    )
if __name__ == "__main__":
   main()
