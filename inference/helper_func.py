import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt


path_ssa = '/scratch-second/radjoe_data/Preprocess_BraTS2023_SSA_Training/images_UNN'
path_gli = '/scratch-second/radjoe_data/Preprocess_BraTS2023_GLI_Training/images_UNN'

save_path_ssa = '/scratch-second/radjoe_data/JPG_BraTS_SSA'
save_path_gli = '/scratch-second/radjoe_data/JPG_BraTS_GLI'

def convert_to_jpg(datapath, savepath):
    for img in os.listdir(datapath):
        if img.endswith('.npy'):
            data_name = img.split('-')[0:4]
            data_name = '-'.join(data_name)
            data_dir = f'{savepath}/{data_name}'
            os.makedirs(data_dir, exist_ok=True)
            # print(data_dir)


            load_path = os.path.join(datapath, img)
            arr = np.load(load_path)
            arr = arr[1]

            for idx in range(arr.shape[2]):
                arr_slice = arr[:,:,idx]
                print(arr_slice.shape)

                arr_norm = arr_slice - np.min(arr_slice)
                if np.max(arr_norm) > 0:
                    arr_norm = (arr_norm / np.max(arr_norm)) * 255
                else:
                    arr_norm = np.zeros_like(arr_slice)
                arr_norm = arr_norm.astype(np.uint8)
                
                im = Image.fromarray(arr_norm)

                save_path = os.path.join(data_dir,f'{idx}.jpeg')
                print(save_path)
                im.save(save_path)
            
            
# def overlay_mask(mask, image):



# convert_to_jpg(path_ssa,save_path_ssa)
# convert_to_jpg(path_gli,save_path_gli)

# img = cv2.imread('/scratch-second/radjoe_data/JPG_BraTS/BraTS-SSA-00096/153.jpeg') 

# cv2.imshow('image', img)
# plt.show()


