# Create Train and Validation Directory
import os
import shutil
from sklearn.model_selection import train_test_split



img_path = '/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/images_UNN'
label_path = '/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN'
save_path = '/scratch_net/ken/radjoe/BraTS_SSA'

def folder_split(folder_path, label_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    list_files = os.listdir(folder_path)

    list_files = [f for f in list_files if f.endswith('.npy')]
    train_files, cross_val = train_test_split(list_files, test_size= 50, random_state=42)

    label_files = []
    for lbl in os.listdir(label_path):
        for img in cross_val:
            print(img[:-8], lbl[:-8])
            if img[:-8] == lbl[:-8]:
                label_files.append(lbl)

    trainlabel_files = []
    for lbl in os.listdir(label_path):
        for img in train_files:
            print(img[:-8], lbl[:-8])
            if img[:-8] == lbl[:-8]:
                trainlabel_files.append(lbl)

    val_savepath = f'{save_path}/Test/images_UNN'
    label_savepath = f'{save_path}/Test/labels_UNN'
    train_savepath = f'{save_path}/Validation/images_UNN'
    trainlabel_savepath = f'{save_path}/Validation/labels_UNN'
    os.makedirs(val_savepath, exist_ok=True)
    os.makedirs(label_savepath, exist_ok=True)
    os.makedirs(train_savepath, exist_ok=True)
    os.makedirs(trainlabel_savepath, exist_ok=True)

    for file in cross_val:
        shutil.copy(os.path.join(folder_path, file), val_savepath)

    for file in label_files:
        shutil.copy(os.path.join(label_path, file), label_savepath)

    for file in train_files:
        shutil.copy(os.path.join(folder_path, file), train_savepath)

    for file in trainlabel_files:
        shutil.copy(os.path.join(label_path, file), trainlabel_savepath)

    # for folder in train_files:
    #     shutil.copytree(os.path.join(folder_path, folder), os.path.join(train_savepath, folder))


folder_split(img_path, label_path, save_path)