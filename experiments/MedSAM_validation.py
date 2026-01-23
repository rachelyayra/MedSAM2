## Predict And Save Prediction in Directory

# Import Relevant Libraries
import os
import sys
sys.path.append('/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/')

from datasets.datasets import MRIDataset, DRIVEDataset, STAREDataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from models.MedSAM2 import MedSAM2Model

import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from copy import deepcopy

import argparse
import torch

from monai.metrics import DiceMetric,  HausdorffDistanceMetric,  MeanIoU





def postprocess_save(prediction,label_shape, model_type):
    
    if model_type == 'video':
        total_mask = np.zeros((4,192,224, 160))
        print(f'label shape {(total_mask.shape)}')
        for out_frame_idx in range(0, 160):
            for out_obj_id, out_mask in prediction[out_frame_idx].items():
                print(f'The outmask shape{out_mask.shape}')
                print(total_mask.shape)
                total_mask[:,:, :, out_frame_idx] = out_mask
        return total_mask
    else: 
        return prediction



def predict_and_save(model_cfg=None, checkpoint=None, video_path=None, dataloader=None,savepath = None, model_type=None, model=None, dataset_type=None, hd_metric=None, dice_metric=None, mean_iou=None):
    '''
    Dataloader
    Predict
    Save to directory

    '''
    model = MedSAM2Model(model_cfg=model_cfg, checkpoint=checkpoint, model_type=model_type, model=model)
    # print(dataloader)
    inference_time_list = []
    for idx, img_path, label_name, box, point, frames, label in dataloader:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        label_shape = label.shape
        saves =  img_path[0][:-8]
        point = point.squeeze(0)
        print(f'This is the shape of the box: {box.shape}')
        box = box.squeeze(0)

        if model_type == 'video':
            img_path = os.path.join(video_path, img_path[0])
            video_dirs = img_path
        else:
            video_dirs = img_path[0]

        print(f'This is the path to the box {box.shape}')
        
        print(frames)

        point = None
        prediction_box = model.predict(video_dirs, box, frames[0], point)
        ender.record()
        os.makedirs(savepath, exist_ok=True)

        save_path_box = os.path.join(savepath,'boxes')


        conpath = os.path.join(savepath,'boxes_con')
        unconpath = os.path.join(savepath,'boxes_uncon')

        os.makedirs(save_path_box, exist_ok=True)

        os.makedirs(conpath, exist_ok=True)
        os.makedirs(unconpath, exist_ok=True)

        save_path_box_label = os.path.join(save_path_box,f'{label_name[0]}.npy')

        output_box = postprocess_save(prediction_box,label_shape, model_type)

        print(f'The output box {output_box.shape}')
        inference_time = starter.elapsed_time(ender) 
        inference_time_list.append([idx, inference_time])
        output_box = torch.tensor(output_box).unsqueeze(0)

        print(f'Shape of output box {output_box.shape}, and label shape {label.shape}')

        dice_metric(output_box, label)

        hd_metric(output_box, label)

        mean_iou(output_box, label)

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

    checkpoint_list = os.listdir(args.checkpoint)
    checkpoint_list = [os.path.join(args.checkpoint, ckpt) for ckpt in checkpoint_list if ckpt.endswith('.pt')]

    validation_metrics = []
    for i in range(len(checkpoint_list)):
        hd_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95.0,
            reduction="mean"
            )


        dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",          
            ignore_empty=True
            )

        mean_iou = MeanIoU(
            include_background=False, 
            reduction="mean"
            )
        dataloader = DataLoader(dataset, shuffle=False)
        print('Here it is')
        predict_and_save(
            model_cfg=args.model_cfg,
            checkpoint=checkpoint_list[i],
            video_path=args.video_path,
            dataloader=dataloader,
            savepath=args.save_path,
            model_type= args.model_type,
            hd_metric=hd_metric,
            dice_metric=dice_metric,
            mean_iou=mean_iou,
        )
        validation_metrics.append([checkpoint_list[i], dice_metric.aggregate()[0].item(), hd_metric.aggregate()[0].item(), mean_iou.aggregate()[0].item()])

        dice_metric.reset()
        hd_metric.reset()
        mean_iou.reset()
    df = pd.DataFrame(validation_metrics, columns=['Sample', 'Dice', 'Hausdorff', 'Mean IoU'])
    df.to_csv(f'{args.save_path}/validation_metrics.csv', index=False)
if __name__ == "__main__":
   main()

