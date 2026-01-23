# Check IOU, DSC and Haudorf distance
import os
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, compute_generalized_dice
import torch
from monai.metrics import HausdorffDistanceMetric, ConfusionMatrixMetric, MeanIoU
import matplotlib.pyplot as plt
from skimage import measure 

from PIL import Image
import numpy as np


def evaluate_3D(pred_path, label_path, savepath):
    
    hd_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95.0,
    reduction="none"
    )


    dice_metric = DiceMetric(
    include_background=False,
    reduction="none",          
    ignore_empty=True
    )


    precision = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="precision", 
    reduction="none"
    )

    recall = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="recall", 
    reduction="none"
    )


    data_dict = {0: 't2f', 1: 't1n', 2: 't1c', 3: 't2w'}

    df_list = []
    temp_DSC = []
    for i ,sample in enumerate(sorted(os.listdir(pred_path))): 
            temp_DSC.append(sample)
            print(sample)
            try:
                    sample_folder = os.path.join(pred_path, sample)

                    gt_label_name = [i for i in os.listdir(label_path) if (sample[:-4] in i and i.endswith('.npy'))]
                    gt_path = os.path.join(label_path, gt_label_name[0])

                    label_numpy = np.load(gt_path)
                    # print(f'This is label numpy{label_numpy}')
                    label_temp =  torch.from_numpy(label_numpy).unsqueeze(0)

                    pred_numpy = np.load(sample_folder)
                    # pred_numpy = np.transpose(pred_numpy, (1, 2, 3, 0))
                    pred_tensor = torch.from_numpy(pred_numpy).long()
                    

                    # print(f'This is the shape of {label_temp.shape}')
        
                    # background = torch.zeros(pred_tensor.shape[1:]).unsqueeze(0)
                    pred_tensor = pred_tensor.unsqueeze(0)
                    print(f'This is the shape of background {pred_tensor.shape , label_temp.shape}')

                    dice_metric(pred_tensor, label_temp)
                    # dice_metric_mean(pred_tensor, label_temp)
                    hd_metric(pred_tensor, label_temp)
                    precision(pred_tensor, label_temp)

                    recall(pred_tensor, label_temp)

                    print(f'Recall {recall.aggregate()[0]}')
                    df_list.append([sample, float(dice_metric.aggregate()[0][0]), float(dice_metric.aggregate()[0][1]), float(dice_metric.aggregate()[0][2]), float(hd_metric.aggregate()[0][0]), float(hd_metric.aggregate()[0][1]), float(hd_metric.aggregate()[0][2]), float(recall.aggregate()[0][0][0]), float(recall.aggregate()[0][0][1]), float(recall.aggregate()[0][0][2]), float(precision.aggregate()[0][0][0]), float(precision.aggregate()[0][0][1]), float(precision.aggregate()[0][0][2])])
                    print([sample, dice_metric.aggregate()[0][1], hd_metric.aggregate()[0][1]])
                    
                    dice_metric.reset()
                    hd_metric.reset()
                    precision.reset()
                    recall.reset()
   
            except Exception as e: 
                print(e)
                pass
            

            try:
                import gc
                del pred_tensor, label_temp, pred_numpy, label_numpy
                gc.collect()
            except:
                pass
    
    df = pd.DataFrame(df_list)
    df.to_csv(savepath)


def evaluate_retinal(pred_path, label_path, savepath):
    
    sensitivity_metric = ConfusionMatrixMetric(
        metric_name="sensitivity",  # or "recall"
        include_background=False,
        reduction="none"
    )


    dice_metric = DiceMetric(
        include_background=False,
        reduction="none",          
        ignore_empty=True
    )


    iou_metric = MeanIoU(
        include_background=False,
        reduction="none"
    )


    df_list = []

    for i ,sample in enumerate(sorted(os.listdir(pred_path))): 
            # temp_DSC.append(sample)
            # print(sample)
            # try:
                    sample_folder = os.path.join(pred_path, sample)

                    gt_label_name = [i for i in os.listdir(label_path) if (sample[:-4] in i)]
                    print(gt_label_name)
                    gt_path = os.path.join(label_path, gt_label_name[0])
                    
                    label_numpy =  Image.open(gt_path)
                    label_numpy = np.array(label_numpy)

                    label_temp =  torch.from_numpy(label_numpy).unsqueeze(0)
                    label_temp = (label_temp == 255).int()
                    background = (label_temp == 0).int()
                    label_temp = torch.stack([background, label_temp], dim=1)
                    

                    pred_numpy = np.load(sample_folder)
                    print(sample_folder)
                    print(f'shape of predict tensor: {pred_numpy.shape}')
                    background_mask = (pred_numpy == 0).astype(int)
                    background_mask =  torch.from_numpy(background_mask)

                    pred_tensor = torch.from_numpy(pred_numpy).long()
                    pred_tensor = torch.stack([background_mask, pred_tensor], dim=1)
                    
                    

                    print(f'This is the shape of {label_temp.shape}')
        
                    # background = torch.zeros(pred_tensor.shape[1:]).unsqueeze(0)
                    # pred_tensor = torch.cat((background, pred_tensor), dim=1).unsqueeze(0)
                    print(f'This is the shape of background {pred_tensor.shape}')

                    dice_metric(pred_tensor, label_temp)
                    iou_metric(pred_tensor, label_temp)
                    sensitivity_metric(pred_tensor, label_temp)
                    # hd_metric_mean(pred_tensor, label_temp)
                    df_list.append([sample, float(dice_metric.aggregate()[0]), float(iou_metric.aggregate()[0]), float(sensitivity_metric.aggregate()[0])])
                    

                    dice_metric.reset()
                    iou_metric.reset()
                    sensitivity_metric.reset()
            # except Exception as e: 
            #     print(e)
            #     pass
            
            # break
                    try:
                        import gc
                        del pred_tensor, label_temp, pred_numpy, label_numpy
                        gc.collect()
                    except:
                        pass
    
    df = pd.DataFrame(df_list)
    df.to_csv(savepath)

def evaluate_4D(pred_path, label_path, savepath):
    
    hd_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95.0,
    reduction="none"
    )


    dice_metric = DiceMetric(
    include_background=False,
    reduction="none",          
    ignore_empty=True
    )


    precision = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="precision", 
    reduction="none"
    )

    recall = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="recall", 
    reduction="none"
    )


    data_dict = {0: 't2f', 1: 't1n', 2: 't1c', 3: 't2w'}

    df_list = []
    temp_DSC = []
    for i ,sample in enumerate(sorted(os.listdir(pred_path))): 
            temp_DSC.append(sample)
            print(sample)
            try:
                    sample_folder = os.path.join(pred_path, sample)

                    gt_label_name = [i for i in os.listdir(label_path) if (sample[:-4] in i and i.endswith('.npy'))]
                    gt_path = os.path.join(label_path, gt_label_name[0])

                    label_numpy = np.load(gt_path)
                    # print(f'This is label numpy{label_numpy.shape}')
                    label_temp =  torch.from_numpy(label_numpy).unsqueeze(0)

                    pred_numpy = np.load(sample_folder)
                    # pred_numpy = np.transpose(pred_numpy, (1, 2, 3, 0))
                    pred_tensor = torch.from_numpy(pred_numpy).long()
                    

                    print(f'This is the shape of {label_temp.shape}')
        
                    
                    sum_pred = torch.sum(pred_tensor, dim=0)
                    sum_pred = sum_pred < 1
                    sum_pred = sum_pred.unsqueeze(0)
                    print(f'This is the shape of {sum_pred.shape, pred_tensor.shape}')
                    pred_tensor = torch.cat((sum_pred, pred_tensor), dim=0)

                    pred_tensor = pred_tensor.unsqueeze(0)
                    print(f'This is the shape of background {pred_tensor.shape , label_temp.shape}')

                    dice_metric(pred_tensor, label_temp)
                    # dice_metric_mean(pred_tensor, label_temp)
                    hd_metric(pred_tensor, label_temp)
                    precision(pred_tensor, label_temp)

                    recall(pred_tensor, label_temp)

                    print(f'Recall {recall.aggregate()[0]}')
                    df_list.append([sample, float(dice_metric.aggregate()[0][0]), float(dice_metric.aggregate()[0][1]), float(dice_metric.aggregate()[0][2]), float(hd_metric.aggregate()[0][0]), float(hd_metric.aggregate()[0][1]), float(hd_metric.aggregate()[0][2]), float(recall.aggregate()[0][0][0]), float(recall.aggregate()[0][0][1]), float(recall.aggregate()[0][0][2]), float(precision.aggregate()[0][0][0]), float(precision.aggregate()[0][0][1]), float(precision.aggregate()[0][0][2])])
                    print([sample, dice_metric.aggregate()[0][1], hd_metric.aggregate()[0][1]])
                    
                    dice_metric.reset()
                    hd_metric.reset()
                    precision.reset()
                    recall.reset()
   
            except Exception as e: 
                print(e)
                pass
            

            try:
                import gc
                del pred_tensor, label_temp, pred_numpy, label_numpy
                gc.collect()
            except:
                pass
    
    df = pd.DataFrame(df_list)
    df.to_csv(savepath)


def evaluate_3D_CC(pred_path, label_path, savepath):
    
    hd_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95.0,
    reduction="none"
    )


    dice_metric = DiceMetric(
    include_background=False,
    reduction="none",          
    ignore_empty=True
    )


    precision = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="precision", 
    reduction="none"
    )

    recall = ConfusionMatrixMetric(
    include_background=False, 
    metric_name="recall", 
    reduction="none"
    )


    data_dict = {0: 't2f', 1: 't1n', 2: 't1c', 3: 't2w'}

    df_list = []
    temp_DSC = []
    for i ,sample in enumerate(sorted(os.listdir(pred_path))): 
            temp_DSC.append(sample)
            print(sample)
            try:
                    sample_folder = os.path.join(pred_path, sample)

                    gt_label_name = [i for i in os.listdir(label_path) if (sample[:-4] in i and i.endswith('.npy'))]
                    gt_path = os.path.join(label_path, gt_label_name[0])

                    label_numpy = np.load(gt_path)
                    # print(f'This is label numpy{label_numpy}')
                    label_temp =  torch.from_numpy(label_numpy).unsqueeze(0)

                    pred_numpy = np.load(sample_folder)
                    
                    pred_numpy = CC_func(pred_numpy)
                    pred_tensor = torch.from_numpy(pred_numpy).long()
                    

                    # print(f'This is the shape of {label_temp.shape}')
        
                    # background = torch.zeros(pred_tensor.shape[1:]).unsqueeze(0)
                    pred_tensor = pred_tensor.unsqueeze(0)
                    print(f'This is the shape of background {pred_tensor.shape , label_temp.shape}')

                    dice_metric(pred_tensor, label_temp)
                    # dice_metric_mean(pred_tensor, label_temp)
                    hd_metric(pred_tensor, label_temp)
                    precision(pred_tensor, label_temp)

                    recall(pred_tensor, label_temp)

                    print(f'Recall {recall.aggregate()[0]}')
                    df_list.append([sample, float(dice_metric.aggregate()[0][0]), float(dice_metric.aggregate()[0][1]), float(dice_metric.aggregate()[0][2]), float(hd_metric.aggregate()[0][0]), float(hd_metric.aggregate()[0][1]), float(hd_metric.aggregate()[0][2]), float(recall.aggregate()[0][0][0]), float(recall.aggregate()[0][0][1]), float(recall.aggregate()[0][0][2]), float(precision.aggregate()[0][0][0]), float(precision.aggregate()[0][0][1]), float(precision.aggregate()[0][0][2])])
                    print([sample, dice_metric.aggregate()[0][1], hd_metric.aggregate()[0][1]])
                    
                    dice_metric.reset()
                    hd_metric.reset()
                    precision.reset()
                    recall.reset()
   
            except Exception as e: 
                print(e)
                pass
            

            try:
                import gc
                del pred_tensor, label_temp, pred_numpy, label_numpy
                gc.collect()
            except:
                pass
    
    df = pd.DataFrame(df_list)
    df.to_csv(savepath)

def CC_func(prediction):

    C, H, W, D = prediction.shape
    # min_voxels = 100  # adjust for expected object size
    filtered_vol = np.zeros_like(prediction)

    for c in range(C):
        mask = prediction[c].astype(np.uint8)  # binary mask for class c
        min_voxels = mask.sum() * 0.2
        
        # 3D connected component labeling
        labeled_mask = measure.label(mask, connectivity=3)
        
        # Filter small components
        filtered_mask = np.zeros_like(mask)
        for i in range(1, labeled_mask.max() + 1):
            if np.sum(labeled_mask == i) >= min_voxels:
                filtered_mask[labeled_mask == i] = 1
        
        filtered_vol[c] = filtered_mask

    return filtered_vol