## Predict And Save Prediction in Directory

# Import Relevant Libraries
import os
import sys
sys.path.append('/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/')

from dataloaders.data_class import MRIDataset
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.SAM2 import SAM2Model
from models.MedSAM2 import MedSAM2Model
import numpy as np
import cv2
from torch.utils.data import DataLoader
from copy import deepcopy


# Load BraTS GLI and BraTS SSA
# GLI_datapath = '/scratch-second/radjoe_data/Preprocess_BraTS2023_GLI_Training'
# GLI_dataset = MRIDataset(GLI_datapath)

SSA_label_path = '/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN'
SSA_video_path = '/scratch_net/ken/radjoe/JPG_BraTS_SSA'
save_path = '/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/BraTS_SSA/SAM2'

GLI_label_path = '/scratch_net/ken/radjoe/Preprocess_BraTS2023_GLI_Training/labels_UNN'
GLI_video_path = '/scratch_net/ken/radjoe/JPG_BraTS_GLI'
GLI_save_path = '/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/BraTS_GLI/SAM2'

SSA_dataset = MRIDataset(SSA_video_path, SSA_label_path)
GLI_dataset = MRIDataset(GLI_video_path, GLI_label_path)

SSA_dataloader = DataLoader(SSA_dataset, shuffle=False)
GLI_dataloader = DataLoader(GLI_dataset, shuffle=False)

model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'
checkpoint = '/scratch_net/ken/radjoe/Projects/Experiments/SAM_DIR/sam2/checkpoints/sam2.1_hiera_tiny.pt' 


# Predict and save in directory


def save_to_directory(tensor, datapath):
    np.save(datapath, tensor)


def predict_and_save(model_cfg, checkpoint,datapath, video_path, dataloader,savepath):
    '''
    Dataloader
    Predict
    Save to directory
    '''
    model = SAM2Model(model_cfg, checkpoint)
    med_model = MedSAM2Model(model_cfg, checkpoint)

    for img_path, label_name, box, frames, label in dataloader:
        # print(f'This is the path to the image {box}')
        # break
        print(label.shape, box, frames)
        img_path_copy = deepcopy(img_path[0])
        img_path = os.path.join(video_path, img_path[0])
        try:
            prediction = model.predict(img_path, box, frames)
            prediction_medsam2 = med_model.predict(img_path, box, frames)

        except:
            continue
        data_dir = os.path.join(datapath,label_name[0])
        # os.makedirs(data_dir, exist_ok=True)
        # print(f'This is the path to the label {data_dir}')

        total_mask = []

        for out_frame_idx in range(0, 160):

            os.makedirs(savepath, exist_ok=True)
            
            save_path = os.path.join(savepath,'np_files')
            os.makedirs(save_path, exist_ok=True)
            print(save_path)
            save_path = os.path.join(save_path,f'{label_name[0]}.npy')
            out_temp = []

            for out_obj_id, out_mask in prediction[out_frame_idx].items():
                out_temp.append(out_mask[0])
                print(out_mask.shape)

            img = cv2.imread(os.path.join(img_path, f'{out_frame_idx}.jpeg'))


            out_mask1 = out_temp[0]
            out_mask2 = out_temp[1]
            out_mask3 = out_temp[2]



            out_mask1 = (out_mask1 > 0).astype(np.uint8)
            out_mask2 = (out_mask2 > 0).astype(np.uint8)
            out_mask3 = (out_mask3 > 0).astype(np.uint8)

            stacked = np.stack([out_mask1, out_mask2, out_mask3], axis=0)

            Height, Weight = out_mask1.shape
            colored_out_mask = np.zeros((Height, Weight, 3), dtype=np.uint8)

            color1 = [255, 0, 0]     # Red
            color2 = [255, 255, 0]   # Yellow
            color3 = [128, 0, 128]   # Purple

            colored_out_mask[out_mask1 == 1] = color1
            colored_out_mask[out_mask2 == 1] = color2
            colored_out_mask[out_mask3 == 1] = color3


            total_mask.append(stacked)
            os.makedirs(f'{savepath}/masks/{img_path_copy}', exist_ok=True)
            cv2.imwrite(f'{savepath}/masks/{img_path_copy}/{out_frame_idx}.jpg', colored_out_mask)
            mask_out = cv2.imread(f'{savepath}/masks/{img_path_copy}/{out_frame_idx}.jpg')
            mask_img = cv2.addWeighted(img, 0.7, mask_out, 0.3, 0)
            mask_img = cv2.rectangle(mask_img,(int(box[0][0][0][0]), int(box[0][0][0][1])), (int(box[0][0][0][2]), int(box[0][0][0][3])),(255, 0, 0), 1)
            mask_img = cv2.rectangle(mask_img,(int(box[1][0][0][0]), int(box[1][0][0][1])), (int(box[1][0][0][2]), int(box[1][0][0][3])),(0, 255, 0), 1)
            mask_img = cv2.rectangle(mask_img,(int(box[2][0][0][0]), int(box[2][0][0][1])), (int(box[2][0][0][2]), int(box[2][0][0][3])),(0, 0, 255), 1)
            cv2.imwrite(f'{savepath}/masks/{img_path_copy}/{out_frame_idx}.jpg', mask_img)

            print(img.shape)


            mask1 = label[0, 0,:,:,out_frame_idx].numpy()
            mask2 = label[0, 1,:,:,out_frame_idx].numpy()
            mask3 = label[0, 2,:,:,out_frame_idx].numpy()

            mask1 = (mask1 > 0).astype(np.uint8)
            mask2 = (mask2 > 0).astype(np.uint8)
            mask3 = (mask3 > 0).astype(np.uint8)

            Height, Weight = mask1.shape
            colored_mask = np.zeros((Height, Weight, 3), dtype=np.uint8)

            color1 = [255, 0, 0]     # Red
            color2 = [255, 255, 0]   # Yellow
            color3 = [128, 0, 128]   # Purple

            colored_mask[mask1 == 1] = color1
            colored_mask[mask2 == 1] = color2
            colored_mask[mask3 == 1] = color3

            os.makedirs(f'{savepath}/true_masks/{img_path_copy}', exist_ok=True)
            print(f'True Mask Dir: {savepath}/true_masks/{img_path_copy}')
            cv2.imwrite(f'{savepath}/true_masks/{img_path_copy}/{out_frame_idx}.jpg', colored_mask)
            mask = cv2.imread(f'{savepath}/true_masks/{img_path_copy}/{out_frame_idx}.jpg')
            true_mask = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
            true_mask = cv2.rectangle(true_mask,(int(box[0][0][0][0]), int(box[0][0][0][1])), (int(box[0][0][0][2]), int(box[0][0][0][3])),(255, 0, 0), 1)
            true_mask = cv2.rectangle(true_mask,(int(box[1][0][0][0]), int(box[1][0][0][1])), (int(box[1][0][0][2]), int(box[1][0][0][3])),(255, 0, 0), 1)
            true_mask = cv2.rectangle(true_mask,(int(box[2][0][0][0]), int(box[2][0][0][1])), (int(box[2][0][0][2]), int(box[2][0][0][3])),(255, 0, 0), 1)
            cv2.imwrite(f'{savepath}/true_masks/{img_path_copy}/{out_frame_idx}.jpg', true_mask)

        total_mask = np.stack(total_mask)
        print(save_path)
        np.save(save_path, total_mask)

            

        
        


predict_and_save(model_cfg=model_cfg, checkpoint=checkpoint, datapath=save_path, video_path=SSA_video_path, dataloader=SSA_dataloader, savepath=save_path)

predict_and_save(model_cfg=model_cfg, checkpoint=checkpoint, datapath=GLI_save_path, video_path=GLI_video_path, dataloader=GLI_dataloader, savepath=GLI_save_path)