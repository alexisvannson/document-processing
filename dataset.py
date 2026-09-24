import torch
import os
import cv2
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.filters import threshold_sauvola
from torchvision.io import decode_image

from getDBprobMap import generate_dbnet_prob_map, generate_dbnet_thresh_map

"""window_size (Odd Integer): Determines the size of the local neighborhood used to calculate the threshold. Larger windows (e.g., 25, 35) are better for capturing larger text or thicker features, while smaller windows (e.g., 11, 15) pick up fine, thin details but are more sensitive to background noise.

k (Float): Controls the threshold's sensitivity to local variance. The standard default is 0.2. If your output is too noisy (background artifacts showing up as text), increase k (e.g., 0.3 or 0.4). If faint text is being lost, decrease k (e.g., 0.1).
    """

def sauvola_fn(image, **kwargs):
    # Albumentations passes a numpy array. Ensure it is grayscale.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Calculate and apply threshold
    thresh = threshold_sauvola(gray, window_size=25, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    
    # If the rest of your model expects 3 channels (e.g., ResNet), duplicate the grayscale channel
    if len(image.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
    return binary

def apply_illumination_gradient(image, **kwargs):
    """
    Applies a synthetic, randomized linear illumination gradient across the image.
    Uses purely vectorized NumPy for maximum throughput.
    """
    h, w = image.shape[:2]
    
    # Generate coordinate grids
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Randomize gradient direction (angle) and intensity (darkness of the shadow)
    angle = np.random.uniform(0, 2 * np.pi)
    min_brightness = np.random.uniform(0.3, 0.7) # Darkest region retains 30%-70% light
    
    # Calculate directional gradient mask
    gradient = X * np.cos(angle) + Y * np.sin(angle)
    
    # Normalize gradient to 0.0 - 1.0
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
    
    # Scale mask to range [min_brightness, 1.0]
    mask = min_brightness + (1.0 - min_brightness) * gradient
    
    # Broadcast mask for RGB channels if necessary
    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        
    # Apply and return as fast uint8
    return np.clip(image * mask, 0, 255).astype(np.uint8)


# Geometric transforms are applied to the image AND the target maps (passed as `masks=`),
# so the labels stay aligned. Padding is filled with 0, which also zeroes prob_mask there
# (padded pixels are ignored by the loss).
# No Sauvola here: binarization is what DBNet learns, so feed it the real image.
train_transform = A.Compose([
    # Spatial / Geometric Distortions
    A.LongestMaxSize(max_size=1280),  # same working scale as validation (some photos are 4096px)
    A.RandomScale(scale_limit=(-0.5, 0.0), p=1.0),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7), # Small rotations
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    # Crop around a random text pixel (reads `mask=gt_prob`) so crops don't land on background
    A.CropNonEmptyMaskIfExists(height=640, width=640, p=1.0),
    A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, p=0.5),

    # Lighting / Color Distortions (image only)
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=5, p=0.5),
    A.Lambda(name="IlluminationGradient", image=apply_illumination_gradient, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

    # PyTorch Formatting
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

transform = A.Compose([
    A.LongestMaxSize(max_size=1280),
    # DBNet needs H and W divisible by 32 (backbone stride). Pad bottom/right only so
    # box coordinates only need rescaling (by 1280 / longest side) to match the input.
    A.PadIfNeeded(
        min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32,
        position="top_left", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class RobustVisionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Read with OpenCV (fastest) - returns BGR
        image = cv2.imread(self.image_paths[idx])
        
        # 2. Convert to standard RGB 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Apply the heavily optimized pipeline
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Returning a dummy label of 0 for demonstration
        return image, 0
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class DBNetDataset(Dataset):
    """
    Receipt text detection dataset for DBNet.
    df: rows of dataset_receipt/metadata.pkl (file_name, bboxes with x1..x4 / y1..y4 corners)
    Returns: image, gt_prob, prob_mask, gt_thresh, thresh_mask  (maps are float tensors of shape 1xHxW)
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(os.path.join(self.img_dir, row.file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Corners are ordered top-left, top-right, bottom-right, bottom-left
        polygons = [[[box[f"x{k}"], box[f"y{k}"]] for k in range(1, 5)] for box in row.bboxes]

        gt_prob = generate_dbnet_prob_map(image.shape, polygons)
        gt_thresh, thresh_mask = generate_dbnet_thresh_map(image.shape, polygons)
        prob_mask = np.ones(image.shape[:2], dtype=np.float32)  # 0 = ignored by the loss

        maps = [gt_prob, prob_mask, gt_thresh, thresh_mask]
        if self.transform:
            # gt_prob goes in `mask` so CropNonEmptyMaskIfExists crops around text only
            augmented = self.transform(image=image, mask=gt_prob, masks=maps[1:])
            image, maps = augmented["image"], [augmented["mask"], *augmented["masks"]]

        maps = [torch.as_tensor(m, dtype=torch.float32).unsqueeze(0) for m in maps]
        return (image, *maps)


def get_dataloaders(metadata_path="dataset_receipt/metadata.pkl", img_dir="dataset_receipt/images",
                    batch_size=8, num_workers=4):
    """dev split -> training, test split -> validation (full-size images, batch_size=1)."""
    df = pd.read_pickle(metadata_path)
    train_dataset = DBNetDataset(df[df.split_origin == "dev"], img_dir, train_transform)
    valid_dataset = DBNetDataset(df[df.split_origin == "test"], img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=num_workers)
    return train_loader, valid_loader
