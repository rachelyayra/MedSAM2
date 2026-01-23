import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt


path_ssa = '/scratch/radjoe/Preprocess_BraTS2023_SSA_Training/images_UNN'
path_gli = '/scratch/radjoe/Preprocess_BraTS2023_GLI_Training/images_UNN'

save_path_ssa = '/scratch/radjoe/JPG_BraTS_SSA'
save_path_gli = '/scratch/radjoe/JPG_BraTS_GLI'


def convert_to_jpg(datapath, savepath):
    os.makedirs(datapath, exist_ok=True)
    for img in sorted(os.listdir(datapath)):
        
        if img.endswith('.npy'):
            print(img)
            data_name = img.split('-')[0:4]
            data_name = '-'.join(data_name)
            data_dir = f'{savepath}/{data_name}'
            os.makedirs(data_dir, exist_ok=True)
            t2f_dir = f'{data_dir}/t2f'
            t1n_dir = f'{data_dir}/t1n'
            t1c_dir = f'{data_dir}/t1c'
            t2w_dir = f'{data_dir}/t2w'
            os.makedirs(t2f_dir, exist_ok=True)
            os.makedirs(t1n_dir, exist_ok=True)
            os.makedirs(t1c_dir, exist_ok=True)
            os.makedirs(t2w_dir, exist_ok=True)
            print(data_dir)

            data_dict = {0: t2f_dir, 1: t1n_dir, 2: t1c_dir, 3: t2w_dir}

            load_path = os.path.join(datapath, img)
            arr = np.load(load_path)
            print(np.unique(arr))


def convert_img_folder_to_jpg(datapath, savepath):
    os.makedirs(savepath, exist_ok=True)
    for video in sorted(os.listdir(datapath)):
        spath = os.path.join(savepath,video)
        lpath = os.path.join(datapath,video)
        os.makedirs(spath, exist_ok=True)
        list_frames = sorted(os.listdir(lpath))
        for idx in range(len(list_frames)):
            fpath = os.path.join(lpath, list_frames[idx])
            # Load PNG image
            img = Image.open(fpath)

            # Convert to RGB (PNG may have alpha channel which JPEG doesn't support)
            img = img.convert("RGB")

            # Save as JPEG
            img.save(f'{spath}/{idx}.jpeg', "JPEG", quality=95)



# convert_to_jpg(path_ssa,save_path_ssa)
# # convert_to_jpg(path_gli,save_path_gli)

# # img = cv2.imread('/scratch-second/radjoe_data/JPG_BraTS/BraTS-SSA-00096/153.jpeg') 

# # cv2.imshow('image', img)
# # plt.show()
ac_dc_loadpath = '/scratch_net/ken/radjoe/rgb_anon/night/val_ref'
ac_dc_savepath = '/scratch_net/ken/radjoe/jpeg_anon/night/val_ref'
convert_img_folder_to_jpg(ac_dc_loadpath, ac_dc_savepath)

