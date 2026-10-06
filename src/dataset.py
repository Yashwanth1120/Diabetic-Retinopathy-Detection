"""
dataset.py
----------
Loads the APTOS 2019 retinal image dataset, applies preprocessing/augmentation,
and provides a WeightedRandomSampler to deal with class imbalance (the dataset
has far more "No DR" images than severe cases).
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


class APTOSDataset(Dataset):
    """
    Expects:
        data_dir/train.csv          columns: id_code, diagnosis
        data_dir/train_images/*.png
    """

    def __init__(self, data_dir, csv_file="train.csv", image_dir="train_images",
                 transform=None, binary=False, indices=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        df = pd.read_csv(os.path.join(data_dir, csv_file))

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        image = Image.open(img_path).convert("RGB")

        label = int(row["diagnosis"])
        if self.binary:
            # 0 = No DR, 1 = any level of DR present
            label = 0 if label == 0 else 1

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        if self.binary:
            return self.df["diagnosis"].apply(lambda x: 0 if x == 0 else 1).values
        return self.df["diagnosis"].values


def get_transforms(train=True, img_size=224):
    """Standard ImageNet-style preprocessing, with augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


def get_weighted_sampler(dataset):
    """
    Builds a WeightedRandomSampler so that rare classes (e.g. Severe,
    Proliferative DR) are sampled roughly as often as common ones during
    training. This matters a lot here because the raw dataset is skewed
    heavily toward "No DR".
    """
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(data_dir, batch_size=32, img_size=224, binary=False,
                     val_split=0.15, seed=42):
    """
    Splits the dataset into train/val and returns DataLoaders.
    Training loader uses the weighted sampler; validation loader does not
    (we want validation metrics to reflect the real class distribution).
    """
    full_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    n = len(full_df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_size = int(n * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ds = APTOSDataset(data_dir, transform=get_transforms(train=True, img_size=img_size),
                             binary=binary, indices=train_indices)
    val_ds = APTOSDataset(data_dir, transform=get_transforms(train=False, img_size=img_size),
                           binary=binary, indices=val_indices)

    sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 2 if binary else 5
    return train_loader, val_loader, num_classes
