from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Helper Functions
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats 
from scipy.stats import shapiro , kstest, mannwhitneyu, ttest_rel
import torch
from torch import nn
from datasets.datasets import MRIDataset
from matplotlib.colors import ListedColormap
np.random.seed(42)
from sklearn.manifold import TSNE



def clean_df(df, fused=False, data_type='retina'):
    df = df.drop(columns = [df.columns[0]])
    if data_type == 'retina':
        df.columns = ['name', 'dice', 'IOU', 'Sensitivity']
    else:
        df.columns = ['name', 'dice_NCR', 'dice_ED', 'dice_ET', 'hausdorff_NCR', 'hausdorff_ED', 'hausdorff_ET', 'recall_NCR', 'recall_ED', 'recall_ET', 'precision_NCR', 'precision_ED', 'precision_ET']
    df = df.dropna()
    df = df.reset_index(drop=True)
    if fused:
        return df
    df_t2f = df.iloc[::4, :].reset_index(drop=True)
    df_t1n = df.iloc[1::4, :].reset_index(drop=True)
    df_t1c = df.iloc[2::4, :].reset_index(drop=True)
    df_t2w = df.iloc[3::4, :].reset_index(drop=True)
    return df_t2f, df_t1n, df_t1c, df_t2w

def plot_boxplots(A, B):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Dice Scores
    ax[0, 0].boxplot(A[A.columns[1:4]].values)
    ax[0, 0].set_title("SAM2 - Dice")
    ax[0, 0].set_xticklabels(A.columns[1:4], rotation=45)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_ylabel("Dice Score")

    ax[0, 1].boxplot(B[B.columns[1:4]].values)
    ax[0, 1].set_title("MedSAM2 - Dice")
    ax[0, 1].set_xticklabels(B.columns[1:4], rotation=45)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_ylabel("Dice Score")

    # Hausdorff 95
    ax[1, 0].boxplot(A[A.columns[4:]].values)
    ax[1, 0].set_title("SAM2 - Hausdorff95")
    ax[1, 0].set_xticklabels(A.columns[4:], rotation=45)
    ax[1, 0].set_ylabel("Hausdorff95")
    ax[1, 0].set_ylim(0, 200) 

    ax[1, 1].boxplot(B[B.columns[4:]].values)
    ax[1, 1].set_title("MedSAM2 - Hausdorff95")
    ax[1, 1].set_xticklabels(B.columns[4:], rotation=45)
    ax[1, 1].set_ylabel("Hausdorff95")
    ax[1, 1].set_ylim(0, 200)  

    plt.tight_layout()
    plt.show()

def display_image(color, label, b1, frames, class_id, ax):
    colored_out_mask = np.zeros((label.shape[1],label.shape[2] , 3))
    colored_out_mask[label[class_id,:,:,frames] == 1] = color
    ax.imshow(colored_out_mask)
    ax.add_patch(plt.Rectangle(b1[0], b1[2] , -b1[1], ls="--", ec="c", fc="none"))
    

    
def renormalize(arr):
    foreground = arr[arr != 0]
    min_val, max_val = np.min(foreground), np.max(foreground)

    scaled = (arr - min_val) / (max_val - min_val + 1e-8)
    scaled[arr == 0] = 0 

    return scaled

def vis_image(path):
    image_path = path
    imgs = np.load(image_path)
    imgs = np.delete(imgs, 1, axis=0)
    for i in range(imgs.shape[0]):
        imgs[i] = renormalize(imgs[i])
    imgs = np.transpose(imgs, (3, 1, 2,0))
    return imgs


def display_image(color, label, b1, frames, class_id, ax, title):
    
    # print(f'here {label.shape}')
    colored_out_mask = np.zeros((label.shape[1],label.shape[2], 3 ))
    mask = label[class_id, :, :, frames] == 1
    colored_out_mask[mask] = np.array(color, dtype=np.uint8)
    ax.imshow(colored_out_mask)
    # ax.add_patch(plt.Rectangle(b1[0], b1[2] , -b1[1], ls="--", ec="c", fc="none"))
    ax.set_title(title)
def preprocess_bbox(bbox):
    print(bbox)
    x = int(bbox[0][0]) 
    y = int(bbox[0][3]) 
    w = int(bbox[0][2]) - int(bbox[0][0])
    h = int(bbox[0][3]) - int(bbox[0][1])
    return (x,y) , h, w

def fig2img(fig):
    canvas = FigureCanvas(fig)  # Attach the Agg backend
    fig.set_canvas(canvas)      # Set the canvas to the figure
    canvas.draw()
    
    # Get the RGB buffer from the figure
    buf = canvas.buffer_rgba()  # Use buffer_rgba() instead of tostring_rgb
    img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
    return img

def visualize(dataset, pred_path, title, path_img, pred_path2 = None ):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        
        fig, ax = plt.subplots(3, 2)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        fig.suptitle(title)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(pred_path, f'{img[:-8]}.npy')
        # print(read_path)
        pred = np.load(read_path)[1:]
        
        # pred = np.transpose(pred, (3,0,1,2))
        label = np.array(label[1:])
        print(f'label.shape {label.shape}')
        # label = np.transpose(label, (1,2,3,0))


        if pred_path2:
            read_path2 = os.path.join(pred_path2, img[:-8])
            loadpath2 = os.path.join(read_path2, os.listdir(read_path2)[0])
            pred2 = np.load(loadpath2)
            pred2 = np.transpose(pred2, (3,0,1,2))
        
        label_key = ['NCR','ED','ET']

        for i in range(3):
            display_image(colors[i], label, b[i],frames, i, ax[i,0], label_key[i])
            display_image(colors[i], pred, b[i], frames, i, ax[i,1], label_key[i])
            if pred_path2:
                display_image(colors[i], pred2, b[i], frames, i, ax[i,2])

        true_img_path = f'{path_img}/{img}'
        true_img = vis_image(true_img_path)
        plt.show()
        plt.imshow(true_img[frames])
        
            
        plt.show()

def visualize_1d(dataset, pred_path, title, pred_path2 = None, read_direct = False):
    colors = [255, 0, 0]
    
    for i in range(len(dataset)):
        # fig2, ax2 = plt.subplots(1, 2)
        if not pred_path2:
            fig, ax = plt.subplots(3, 2)
        else:
            fig, ax = plt.subplots(3, 3)

        fig.suptitle(title)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        # print(img)
        b1 = preprocess_bbox(bbox=box)


        pred = img

        if pred_path2:
            read_path2 = os.path.join(pred_path2, img[:-8])
            loadpath2 = os.path.join(read_path2, os.listdir(read_path2)[0])
            pred2 = np.load(loadpath2)
            pred2 = np.transpose(pred2, (1,2,3,0))
        
        
        for i in range(len(box)):
            display_image(colors, np.expand_dims( label, axis=0), b1,frames, i, ax[i,0], label_key[i])
            display_image(colors, np.expand_dims( pred, axis=0), b1, frames, i, ax[i,1], label_key[i])
            if pred_path2:
                display_image(colors, pred2, b1, frames, i, ax[i,2])

        true_img_path = f'/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/images_UNN/{img}'
        true_img = vis_image(true_img_path)
        plt.show()
        plt.imshow(true_img[frames])
        
            
        plt.show()

view_transform = A.Compose([
A.RandomBrightnessContrast(p=1.0),
        # A.Normalize(),
        # ToTensorV2(),
    ])

def generate_batch_views(input_image, num_views=4):
    results = []
    for _ in range(num_views):
        augmented = view_transform(image=input_image)
        transformed_image = augmented['image']
        results.append(transformed_image)
    return results
def generate_batch_views(input_image, num_views=4):
    results = []
    for _ in range(num_views):
        augmented = view_transform(image=input_image)
        transformed_image = augmented['image']
        results.append(transformed_image)
    return results

def imshow(img, ax, title=None):
    """Show a tensor or numpy image in matplotlib."""
    # img = np.transpose(img, (2, 0, 1 ))
    # Clip values between 0 and 1

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')

def plot_input_and_views(input_image, augmented_views):
    n_views = len(augmented_views)
    fig, axes = plt.subplots(1, n_views + 1, figsize=(4 * (n_views + 1), 4))
    
    # Show original image (input_image is uint8 HWC, convert to float [0,1])
    axes[0].imshow(input_image.astype(np.float32) / 255.0)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show augmented views
    for i, view in enumerate(augmented_views):
        view = np.transpose(view, (1,2,0))
        imshow(view, axes[i + 1], title=f'Augmented View {i + 1}')
    
    plt.tight_layout()
    plt.show()

def visualize_views(dataset,title, path_img, pred_path2 = None ):

    # graph_num = len(pred_path)
    for i in range(len(dataset)):

        idx, img, label_name, box, points, frames, label = dataset[i]

        true_img_path = f'{path_img}/{img}'
        true_img = vis_image(true_img_path)
        print(true_img[frames].shape)
        true_img = true_img[frames]
        views = generate_batch_views(true_img)
        plot_input_and_views(true_img, views)

        plt.show()




# Example usage:

# input_image = cv2.imread('path/to/image.jpg')
# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


view_transform = A.Compose([
A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ])

def renormalize(arr):
    foreground = arr[arr != 0]
    min_val, max_val = np.min(foreground), np.max(foreground)

    scaled = (arr - min_val) / (max_val - min_val + 1e-8)
    scaled[arr == 0] = 0 

    return scaled

def vis_image(path):
    image_path = path
    imgs = np.load(image_path)
    imgs = np.delete(imgs, 1, axis=0)
    for i in range(imgs.shape[0]):
        imgs[i] = renormalize(imgs[i])
    imgs = np.transpose(imgs, (3, 1, 2,0))
    return imgs

def generate_batch_views(input_image, num_views=4):
    results = []
    for _ in range(num_views):
        augmented = view_transform(image=input_image)
        transformed_image = augmented['image']
        results.append(transformed_image)
    return results

def imshow(img, ax, title=None):
    """Show a tensor or numpy image in matplotlib."""
    # img = np.transpose(img, (2, 0, 1 ))
    # Clip values between 0 and 1

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')

def plot_input_and_views(input_image, augmented_views):
    n_views = len(augmented_views)
    fig, axes = plt.subplots(1, n_views + 1, figsize=(4 * (n_views + 1), 4))
    
    # Show original image (input_image is uint8 HWC, convert to float [0,1])
    axes[0].imshow(input_image.astype(np.float32) / 255.0)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show augmented views
    for i, view in enumerate(augmented_views):
        view = np.transpose(view, (1,2,0))
        imshow(view, axes[i + 1], title=f'Augmented View {i + 1}')
    
    plt.tight_layout()
    plt.show()

def visualize_views(dataset,title, path_img, pred_path2 = None ):

    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        

        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]

        true_img_path = f'{path_img}/{img}'
        true_img = vis_image(true_img_path)
        print(true_img[frames].shape)
        true_img = true_img[frames]
        views = generate_batch_views(true_img)
        plot_input_and_views(true_img, views)
        
            
        plt.show()


def renormalize(arr):
    foreground = arr[arr != 0]
    min_val, max_val = np.min(foreground), np.max(foreground)

    scaled = (arr - min_val) / (max_val - min_val + 1e-8)
    scaled[arr == 0] = 0 

    return scaled

def vis_image(path):
    image_path = path
    imgs = np.load(image_path)
    imgs = np.delete(imgs, 1, axis=0)
    for i in range(imgs.shape[0]):
        imgs[i] = renormalize(imgs[i])
    imgs = np.transpose(imgs, (3, 1, 2,0))
    return imgs

def generate_batch_views(input_image, view_transform, num_views=4, mask=None):
    results = []
    for _ in range(num_views):
        true_img_np = input_image
        label_np = mask.permute(1, 2, 0).cpu().numpy()
        print("Image shape:", true_img_np.shape)  # should be (H, W, C)
        print("Mask shape:", label_np.shape)  # should be (H, W)
        augmented = view_transform(image=true_img_np, label=label_np, seed=42)
        transformed_image = augmented['image']
        results.append(transformed_image)
    return results

def imshow(img, ax, title=None):
    """Show a tensor or numpy image in matplotlib."""
    # img = np.transpose(img, (2, 0, 1 ))
    # Clip values between 0 and 1

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')

def plot_input_and_views(input_image, augmented_views):
    n_views = len(augmented_views)
    fig, axes = plt.subplots(1, n_views + 1, figsize=(4 * (n_views + 1), 4))
    

    axes[0].imshow(input_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    

    for i, view in enumerate(augmented_views):
        view = np.transpose(view, (1,2, 0))
        imshow(view, axes[i + 1], title=f'Augmented View {i + 1}')
    
    plt.tight_layout()
    plt.show()

def visualize_views(dataset,title, path_img,view_transform, pred_path2 = None ):

    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        

        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]

        true_img_path = f'{path_img}/{img}'
        true_img = vis_image(true_img_path)
        # print(true_img[frames].shape)
        true_img = true_img[frames]
        views = generate_batch_views(true_img, view_transform, mask=label[:,:,:, frames])
        plot_input_and_views(true_img, views)
        
            
        plt.show()

def to_numpy_mask(m):
    # torch -> numpy, squeeze channel if present
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    # drop singleton dims to get [H, W]
    if m.ndim == 3 and (m.shape[0] == 1 or m.shape[-1] == 1):
        m = np.squeeze(m)
    # ensure integer type for masks
    if not np.issubdtype(m.dtype, np.integer):
        m = m.astype(np.int32)
    return m

def to_numpy_image(x):
    # expects HWC for Albumentations
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    # if CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1,2,3,4):   # likely CHW
        x = np.transpose(x, (1, 2, 0))
    return x.astype(np.float32)


def visualize_brats(dataset, pred_path, title, path_img, pred_path2 = None ):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        
        fig, ax = plt.subplots(3, 2)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        fig.suptitle(title)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        # print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(path_img, img)
        # print(read_path)
        pred = np.load(read_path)[1:]
        
        # pred = np.transpose(pred, (3,0,1,2))
        label = np.array(label)
        print(f'label.shape {label.shape}')
        # label = np.transpose(label, (1,2,3,0))


        if pred_path2:
            read_path2 = os.path.join(pred_path2, img[:-8])
            loadpath2 = os.path.join(read_path2, os.listdir(read_path2)[0])
            pred2 = np.load(loadpath2)
            pred2 = np.transpose(pred2, (3,0,1,2))
        
        label_key = ['NCR','ED','ET']

        for i in range(3):
            display_image(colors[i], label, b[i],frames, i, ax[i,0], label_key[i])
            display_image(colors[i], pred, b[i], frames, i, ax[i,1], label_key[i])
            if pred_path2:
                display_image(colors[i], pred2, b[i], frames, i, ax[i,2])

        true_img_path = f'{path_img}/{img}'
        true_img = vis_image(true_img_path)
        plt.show()
        plt.imshow(true_img[frames])
        
            
        plt.show()


def visualize_brats(dataset, pred_path, title, path_img):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        
        fig, ax = plt.subplots(3,5)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        # print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(path_img, img)
        # print(read_path)
        pred = np.load(read_path)

        pred = renormalize(pred)
        
                # pred = np.transpose(pred, (3,0,1,2))
        # label: (C, H, W, D) one-hot or per-class masks
        label = np.asarray(label)
        C, H, W, D = label.shape

        # Build a 3D label map (H, W, D)
        label_map = np.zeros((H, W, D), dtype=np.uint8)

        # Fill label_map; use >0 to be robust to {0,255} or float masks
        for c in range(C):
            label_map[label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)

        # Sanity prints per class for the chosen frame
        for c in range(C):
            present_anywhere = np.any(label[c] > 0)
            present_in_frame = np.any(label[c, :, :, frames] > 0)
            print(f"class {c}: any={present_anywhere}, in frame {frames}={present_in_frame}")

        # Colormap with black background (class 0)
        


        # Make colormap with 4 entries (0..3), 0=black
        num_classes = C + 1  # 3 fg + 1 bg
        palette = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 2)))
        palette[0] = [0, 0, 0, 1]  # background black
        cmap = ListedColormap(palette[:num_classes])
        

        # Grayscale inputs
        for k in range(4):
            ax[0, k].imshow(pred[k, :, :, 80], cmap='gray')
            ax[1, k].imshow(pred[k, :, 112, :], cmap='gray')
            ax[2, k].imshow(pred[k, 96, :, :], cmap='gray')
            ax[0, k].axis('off')
            ax[1, k].axis('off')
            ax[2, k].axis('off')


        ax[0, 4].imshow(label_map[:, :, 80], cmap=cmap, interpolation='nearest')
        ax[1, 4].imshow(label_map[:, 112, :], cmap=cmap, interpolation='nearest')
        ax[2, 4].imshow(label_map[96, :, :], cmap=cmap, interpolation='nearest')
        ax[0, 4].axis('off')
        ax[1, 4].axis('off')
        ax[2, 4].axis('off')



        # Titles for each row
        row_titles = ['Axial', 'Coronal', 'Sagittal']

        # Use fig.text with normalized coordinates:
        n_rows = len(row_titles)
        for r, title in enumerate(row_titles):
            # y coordinate: middle of each row, counting from top
            y = 1 - (r + 0.5) / n_rows
            fig.text(0.02, y, title, va='center', ha='center', rotation=90, fontsize=12)

        col_titles = ['Flair', 'T1', 'T2', 'T1c', 'Mask']

        n_cols = len(col_titles)
        plt.tight_layout()

        for c, title in enumerate(col_titles):
            # x = center of each column
            x = (c + 0.5) / n_cols
            # y = little above the top; adjust if you have suptitle
            fig.text(x, 0.99, title, va='bottom', ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'brats_vis_{img}.png', dpi=300)

        plt.show()


def visualize_feats_3D(dataset, pred_path, pca_temp, title, path_img, pred_path2 = None ):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        fig, ax = plt.subplots(1, 2)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(pred_path, f'{img[:-8]}.npy')
        # print(read_path)
        pred = np.load(read_path).squeeze(0)
        pred_fit = pred.reshape(-1, pred.shape[0])

        print(f"pred.shape {pred_fit.shape}")


        # Reduce dimensions with PCA


        # pred = np.transpose(pred, (3,0,1,2))
        pred_label = os.path.join(pred_path2, f'{img[:-8]}.npy')
        pred_label = np.load(pred_label)[1:, :, :, frames]
        torch_pred_label = torch.from_numpy(pred_label)
        torch_pred_label = torch.nn.functional.interpolate(torch_pred_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()
        pred_label = to_numpy_mask(torch_pred_label)
        pred_label_map = np.zeros((pred_label.shape[1], pred_label.shape[2]), dtype=np.uint8)
        for c in range(pred_label.shape[0]):
            pred_label_map[pred_label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"pred_label.shape {pred_label.shape}")

        pred_keep = pred_label_map.flatten() != 0
        pred_lb = pred_label_map.flatten()[pred_keep]
        print(f"pred_keep:{pred_keep.shape}")
        label = np.array(label[1:, :, :, frames])
        torch_label = torch.from_numpy(label)
        torch_label = torch.nn.functional.interpolate(torch_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()

        label = to_numpy_mask(torch_label)
        label_map = np.zeros((label.shape[1], label.shape[2]), dtype=np.uint8)
        for c in range(label.shape[0]):
            label_map[label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"label.shape {label.shape}")
        print(f'label_map.shape {label_map.shape}')

        keep = label_map.flatten() != 0
        lb = label_map.flatten()[keep]
        print(f"Keep:{keep}")
        X = pred_fit[keep]
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        Xz = (X - mu) / sigma

        # 2) optional: per-sample L2 norm (row-wise) for cosine-friendly geometry
        Xn = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-8)
        # pred_fit = pca_temp.transform(pred_fit)
        Z2_tsne = TSNE(n_components=3, perplexity=50, learning_rate='auto', init='pca', n_iter=500, random_state=0).fit_transform(Xn)
        print(f"pred.shape {pred_fit.shape}")

        fig = plt.figure(figsize=(7,6))
        ax3d = fig.add_subplot(111, projection='3d')


        sc = ax3d.scatter(
            Z2_tsne[:, 0],
            Z2_tsne[:, 1],
            Z2_tsne[:, 2],
            c=lb,
            cmap='viridis',
            alpha=0.8,
            s=12
        )

        cb = plt.colorbar(sc, ax=ax3d, pad=0.1)
        cb.set_label('Class')
        cb.set_ticks(sorted(np.unique(lb)))

        ax3d.set_title('PCA of Decoder Features (3D)')
        ax3d.set_xlabel('PCA Component 1')
        ax3d.set_ylabel('PCA Component 2')
        ax3d.set_zlabel('PCA Component 3')
        ax3d.view_init(elev=25, azim=45)  # tweak the view if you like

        plt.tight_layout()
        plt.show()
        fig2 = plt.figure(figsize=(7,6))
        ax3d2 = fig2.add_subplot(111, projection='3d')


        sc2 = ax3d2.scatter(
            pred_fit[pred_keep, 0],
            pred_fit[pred_keep, 1],
            pred_fit[pred_keep, 2],
            c=pred_lb,
            cmap='viridis',
            alpha=0.8,
            s=12
        )

        cb = plt.colorbar(sc2, ax=ax3d2, pad=0.1)
        cb.set_label('Class')
        cb.set_ticks(sorted(np.unique(pred_lb)))

        ax3d2.set_title('PCA of Decoder Features (3D)')
        ax3d2.set_xlabel('PCA Component 1')
        ax3d2.set_ylabel('PCA Component 2')
        ax3d2.set_zlabel('PCA Component 3')
        ax3d2.view_init(elev=25, azim=45)  # tweak the view if you like

        plt.tight_layout()
        plt.show()

def visualize_feats(dataset, pred_path, pca_temp, title, path_img, pred_path2 = None ):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    for i in range(len(dataset)):
        fig1, ax1 = plt.subplots(1, 3)
        fig2, ax2 = plt.subplots(1, 3)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(pred_path, f'{img[:-8]}.npy')
        # print(read_path)
        pred = np.load(read_path).squeeze(0)
        pred_fit = pred.reshape(-1, pred.shape[0])

        print(f"pred.shape {pred_fit.shape}")


        # Reduce dimensions with PCA


        # pred = np.transpose(pred, (3,0,1,2))
        pred_label = os.path.join(pred_path2, f'{img[:-8]}.npy')
        pred_label = np.load(pred_label)[1:, :, :, frames]
        torch_pred_label = torch.from_numpy(pred_label)
        torch_pred_label = torch.nn.functional.interpolate(torch_pred_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()
        pred_label = to_numpy_mask(torch_pred_label)
        pred_label_map = np.zeros((pred_label.shape[1], pred_label.shape[2]), dtype=np.uint8)
        for c in range(pred_label.shape[0]):
            pred_label_map[pred_label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"pred_label.shape {pred_label.shape}")

        pred_keep = pred_label_map.flatten() != 0
        pred_lb = pred_label_map.flatten()[pred_keep]
        print(f"pred_keep:{pred_keep.shape}")
        label = np.array(label[1:, :, :, frames])
        torch_label = torch.from_numpy(label)
        torch_label = torch.nn.functional.interpolate(torch_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()

        label = to_numpy_mask(torch_label)
        label_map = np.zeros((label.shape[1], label.shape[2]), dtype=np.uint8)
        for c in range(label.shape[0]):
            label_map[label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"label.shape {label.shape}")
        print(f'label_map.shape {label_map.shape}')

        keep = label_map.flatten() != 0
        lb = label_map.flatten()[keep]
        print(f"Keep:{keep}")
        X = pred_fit[keep]
        Xp = pred_fit[pred_keep]

        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        Xz = (X - mu) / sigma
        Xn = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-8)

        mu_p = Xp.mean(axis=0, keepdims=True)
        sigma_p = Xp.std(axis=0, keepdims=True) + 1e-8
        Xz_p = (Xp - mu_p) / sigma_p
        Xn_p = Xz_p / (np.linalg.norm(Xz_p, axis=1, keepdims=True) + 1e-8)
        Xn = pca_temp.transform(Xn)
        if len(Xn_p) > 0:
            Xn_p = pca_temp.transform(Xn_p)
        # Z2_tsne = TSNE(n_components=3, perplexity=50, learning_rate='auto', init='pca', n_iter=500, random_state=0).fit_transform(Xn)
        print(f"pred.shape {pred_fit.shape}")



        ax1[0].scatter(
            Xn[:, 0],
            Xn[:, 1],
            c=lb,
            cmap='viridis',
            alpha=0.8,
            s=12
        )
        ax1[0].set_title('t-SNE of Decoder Features (2D)')

        ax1[1].scatter(
            Xn[:, 0],
            Xn[:, 2],
            c=lb,
            cmap='viridis',
            alpha=0.8,
            s=12
        )
        ax1[1].set_title('t-SNE of Decoder Features (2D)')

        ax1[2].scatter(
            Xn[:, 1],
            Xn[:, 2],
            c=lb,
            cmap='viridis',
            alpha=0.8,
            s=12
        )
        ax1[2].set_title('t-SNE of Decoder Features (2D)')

        if len(Xn_p) > 0:
            ax2[0].scatter(
                Xn_p[:, 0],
                Xn_p[:, 1],
                c=pred_lb,
                cmap='viridis',
                alpha=0.8,
                s=12
            )
            ax2[0].set_title('PCA of Decoder Features (3D)')
            ax2[0].set_xlabel('PCA Component 1')
            ax2[0].set_ylabel('PCA Component 2')

            ax2[1].scatter(
                Xn_p[:, 0],
                Xn_p[:, 1],
                c=pred_lb,
                cmap='viridis',
                alpha=0.8,
                s=12
            )
            ax2[1].set_title('PCA of Decoder Features (3D)')
            ax2[1].set_xlabel('PCA Component 1')
            ax2[1].set_ylabel('PCA Component 2')


            ax2[2].scatter(
                Xn_p[:, 0],
                Xn_p[:, 1],
                c=pred_lb,
                cmap='viridis',
                alpha=0.8,
                s=12
            )
            ax2[2].set_title('PCA of Decoder Features (3D)')
            ax2[2].set_xlabel('PCA Component 1')
            ax2[2].set_ylabel('PCA Component 2')

            plt.tight_layout()
            plt.show()


def visualize_class_centroids(dataset, pred_path, pca_temp, title, path_img, pred_path2=None):
    for i in range(len(dataset)):
        fig1, ax1 = plt.subplots(1, 3)
        fig2, ax2 = plt.subplots(1, 3)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(pred_path, f'{img[:-8]}.npy')
        # print(read_path)
        pred = np.load(read_path).squeeze(0)
        pred_fit = pred.reshape(-1, pred.shape[0])

        print(f"pred.shape {pred_fit.shape}")


        # Reduce dimensions with PCA


        # pred = np.transpose(pred, (3,0,1,2))
        pred_label = os.path.join(pred_path2, f'{img[:-8]}.npy')
        pred_label = np.load(pred_label)[1:, :, :, frames]
        torch_pred_label = torch.from_numpy(pred_label)
        torch_pred_label = torch.nn.functional.interpolate(torch_pred_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()
        pred_label = to_numpy_mask(torch_pred_label)
        pred_label_map = np.zeros((pred_label.shape[1], pred_label.shape[2]), dtype=np.uint8)
        for c in range(pred_label.shape[0]):
            pred_label_map[pred_label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"pred_label.shape {pred_label.shape}")

        pred_keep = pred_label_map.flatten() != 0
        pred_lb = pred_label_map.flatten()[pred_keep]
        print(f"pred_keep:{pred_keep.shape}")
        label = np.array(label[1:, :, :, frames])
        torch_label = torch.from_numpy(label)
        torch_label = torch.nn.functional.interpolate(torch_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()

        label = to_numpy_mask(torch_label)
        label_map = np.zeros((label.shape[1], label.shape[2]), dtype=np.uint8)
        for c in range(label.shape[0]):
            label_map[label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"label.shape {label.shape}")
        print(f'label_map.shape {label_map.shape}')

        keep = label_map.flatten() != 0
        lb = label_map.flatten()[keep]
        print(f"Keep:{keep}")
        X = pred_fit[keep]
        Xp = pred_fit[pred_keep]

        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        Xz = (X - mu) / sigma
        Xn = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-8)

        mu_p = Xp.mean(axis=0, keepdims=True)
        sigma_p = Xp.std(axis=0, keepdims=True) + 1e-8
        Xz_p = (Xp - mu_p) / sigma_p
        Xn_p = Xz_p / (np.linalg.norm(Xz_p, axis=1, keepdims=True) + 1e-8)
        Xn = pca_temp.transform(Xn)
        if len(Xn_p) > 0:
            Xn_p = pca_temp.transform(Xn_p)
            centroids_Xn_p = class_centroids(Xn_p, pred_lb)
            print(f"centroids_Xn_p {centroids_Xn_p}")
            D_euc_p = [pairwise_euclidean(x) for x in centroids_Xn_p]
            D_cos_p = [pairwise_cosine(x) for x in centroids_Xn_p]
        print("Euclidean Distance Matrix between Class Centroids:\n", D_euc_p)
        print("Cosine Distance Matrix between Class Centroids:\n", D_cos_p)
        # Z2_tsne = TSNE(n_components=3, perplexity=50, learning_rate='auto', init='pca', n_iter=500, random_state=0).fit_transform(Xn)
        print(f"pred.shape {pred_fit.shape}")

        centroids_Xn = class_centroids(Xn, lb)
        D_euc = [pairwise_euclidean(x) for x in centroids_Xn]
        D_cos = [pairwise_cosine(x) for x in centroids_Xn]
        print("Euclidean Distance Matrix between Class Centroids:\n", D_euc)
        print("Cosine Distance Matrix between Class Centroids:\n", D_cos)

        plot_matrix(D_euc, "Inter-class Euclidean distances", class_names=label_name)

        plt.show()

# X: (N, D) embeddings; y: (N,) integer class labels 0..K-1
def class_centroids(X, y):
    classes = np.unique(y)
    centroids = np.stack([X[y == c][:1].mean(axis=0) for c in classes], axis=0)  # (K, D)
    return classes, centroids

def pairwise_euclidean(A):
    # A: (K, D) -> (K, K)
    aa = (A**2).sum(axis=1, keepdims=True)
    d2 = aa + aa.T - 2 * A @ A.T
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

def pairwise_cosine(A, eps=1e-8):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    sim = A_n @ A_n.T
    # Convert similarity to distance in [0,2]
    return 1.0 - sim

# Example:
# classes, C = class_centroids(X, y)
# D = pairwise_euclidean(C)   # or pairwise_cosine(C)

def plot_matrix(M, title="", class_names=None):
    plt.figure(figsize=(6,5))
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.show()

def downsample_mask_nearest(mask, fy=2, fx=2):
    mask = np.asarray(mask)
    H, W = mask.shape[-2:]
    H2, W2 = H // fy, W // fx
    return mask[..., :H2*fy, :W2*fx][..., ::fy, ::fx]


def prod_feats(dataset, pred_path, pca_temp, title, path_img, pred_path2 = None ):
    colors = [
        [255, 0, 0],       
        [255, 255, 0],     
        [128, 0, 128],     
    ]
    # graph_num = len(pred_path)
    feat_array  = np.empty((0, 32), dtype=np.float32)
    label_array = np.empty((0,),     dtype=np.int32)
    for i in range(len(dataset)):
        fig, ax = plt.subplots(1, 2)
        # print(len(ax))
        idx, img, label_name, box, points, frames, label = dataset[i]
        print(img)
        b1 = preprocess_bbox(bbox=box[0])
        # b2 = preprocess_bbox(bbox=box[1])
        # b3 = preprocess_bbox(bbox=box[2])
        b = [b1,b1, b1]
        
        read_path = os.path.join(pred_path, f'{img[:-8]}.npy')
        # print(read_path)
        pred = np.load(read_path).squeeze(0)
        pred_fit = pred.reshape(-1, pred.shape[0])

        label = np.array(label[1:, :, :, frames])
        torch_label = torch.from_numpy(label)

        torch_label = torch.nn.functional.interpolate(torch_label.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).long()
        label = to_numpy_mask(torch_label)
        label_map = np.zeros((label.shape[1], label.shape[2]), dtype=np.uint8)
        for c in range(label.shape[0]):
            label_map[label[c] > 0] = c + 1   # <-- +1 so class 0 channel becomes id 1 (not 0)
        print(f"label.shape {label.shape}")
        print(f'label_map.shape {label_map.shape}')

        feat_array  = np.vstack([feat_array, pred_fit])
        label_array = np.concatenate([label_array, label_map.flatten()])

    return feat_array, label_array
