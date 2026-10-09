import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image

class MRNetDataset(Dataset):
    """
    MRNet PyTorch Dataset for loading Knee MRI voxel sequences.
    
    Supports:
      - Plane selection: 'axial', 'coronal', 'sagittal'
      - Task selection: 'acl', 'meniscus', 'abnormal'
      - Uniform slice sampling to fit sequence length S (speeds up training on CPU/GPU)
      - Data augmentation & ImageNet normalization
    """
    def __init__(self, root_dir, split='train', plane='axial', task='acl', transform=None, num_slices=16):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.plane = plane
        self.task = task
        self.num_slices = num_slices
        self.transform = transform

        # Path to labels CSV
        csv_path = os.path.join(root_dir, f"{'train' if split in ['train', 'val'] else 'valid'}-{task}.csv")
        df = pd.read_csv(csv_path, header=None, names=['id', 'label'], dtype={'id': str, 'label': int})

        # Split 85% train / 15% internal val if split is 'train' or 'val'
        if split in ['train', 'val']:
            np.random.seed(42)
            perm = np.random.permutation(len(df))
            val_size = int(0.15 * len(df))
            if split == 'val':
                indices = perm[:val_size]
            else:
                indices = perm[val_size:]
            self.labels_df = df.iloc[indices].reset_index(drop=True)
        else:
            self.labels_df = df.reset_index(drop=True)

        # Path to slice directory
        series_split = 'train' if split in ['train', 'val'] else 'valid'
        self.series_dir = os.path.join(root_dir, series_split, plane)

        # Default transforms if none provided
        if self.transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        case_id = str(self.labels_df.iloc[idx]['id']).zfill(4)
        label = self.labels_df.iloc[idx]['label']

        file_path = os.path.join(self.series_dir, f"{case_id}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Series file not found: {file_path}")

        # Load MRI volume: shape (S, H, W)
        array = np.load(file_path)
        total_slices, H, W = array.shape

        # Uniform slice sampling if total_slices > self.num_slices
        if self.num_slices is not None and total_slices > self.num_slices:
            indices = np.linspace(0, total_slices - 1, self.num_slices, dtype=int)
            array = array[indices]
        
        # Process each slice into a 3-channel tensor
        processed_slices = []
        for i in range(len(array)):
            slice_2d = array[i]
            # Repeat grayscale to 3 RGB channels
            slice_rgb = np.stack([slice_2d]*3, axis=-1)  # (H, W, 3)
            slice_tensor = self.transform(slice_rgb)     # (3, H, W)
            processed_slices.append(slice_tensor)

        # Stack into sequence tensor: (S, 3, H, W)
        series_tensor = torch.stack(processed_slices, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return series_tensor, label_tensor, case_id
