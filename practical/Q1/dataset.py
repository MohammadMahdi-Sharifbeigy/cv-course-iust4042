import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import torchvision.transforms.functional as TF
import random

# ==============================================================================
# 1. Dataset Wrapper & Preprocessing (Task 2 & 3)
# ==============================================================================
class OxfordPetBinaryDataset(Dataset):
    """
    Custom wrapper for Oxford-IIIT Pet Dataset to handle:
    - Resizing to 256x256
    - Binary Mask mapping (1 -> Animal, 0 -> Background)
    - Joint Data Augmentations (Horizontal Flip, Rotation, Color Jitter)
    - Normalization
    """
    def __init__(self, root_dir, split='train', is_train=False):
        super().__init__()
        # PyTorch built-in dataset provides 'trainval' and 'test' splits
        base_split = 'trainval' if split in ['train', 'val'] else 'test'
        self.dataset = datasets.OxfordIIITPet(
            root=root_dir, 
            split=base_split, 
            target_types='segmentation', 
            download=True
        )
        self.is_train = is_train
        
        # ImageNet normalization stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Task 2: Resize to 256x256
        image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        # Masks should be resized with Nearest Neighbor to avoid interpolating categorical values
        mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
        
        # Task 3: Data Augmentation (Only during training)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Rotation (-10 to 10 degrees)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            
            # Brightness and Contrast (Only to image)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)

        # Convert to Tensor
        image = TF.to_tensor(image) # Scales to [0, 1]
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        # Task 2: Normalize image
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # Task 2: Binary Segmentation Mask Mapping
        # Original Oxford Pet mask values: 
        # 1: Foreground, 2: Background, 3: Not Classified
        # We need: 1: Animal, 0: Background
        binary_mask = torch.zeros_like(mask)
        binary_mask[mask == 1] = 1 # Animal
        # Values 2 and 3 remain 0 (Background)
        
        return image, binary_mask.unsqueeze(0)

def get_datasets(root_dir='./data'):
    """
    Loads and splits the dataset into train, validation, and test sets.
    """
    # Load the base splits
    trainval_ds = OxfordPetBinaryDataset(root_dir, split='train', is_train=True)
    test_ds = OxfordPetBinaryDataset(root_dir, split='test', is_train=False)
    
    # Task 2: Split 'trainval' into 'train' and 'validation'
    # We'll use an 80/20 split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(trainval_ds))
    val_size = len(trainval_ds) - train_size
    
    train_ds, val_ds = random_split(trainval_ds, [train_size, val_size], generator=generator)
    
    # Important: The validation set shouldn't have random augmentations applied.
    # Since we set is_train=True for the whole trainval_ds, we need to disable it for val_ds.
    # We can do this by creating a fresh dataset instance for val and sharing indices, 
    # but for simplicity in this dataset wrapper, we can just alter the flag for the val subset.
    val_ds.dataset.is_train = False 
    # (Note: Python random_split returns a Subset. The above modifies the underlying dataset flag,
    # which would turn off train augs for train_ds too. Let's fix this.)
    
    # Proper way to handle different transforms for train and val splits:
    train_ds = OxfordPetBinaryDataset(root_dir, split='train', is_train=True)
    val_ds = OxfordPetBinaryDataset(root_dir, split='train', is_train=False)
    
    # Share the split indices
    indices = torch.randperm(len(train_ds), generator=generator).tolist()
    train_ds = torch.utils.data.Subset(train_ds, indices[:train_size])
    val_ds = torch.utils.data.Subset(val_ds, indices[train_size:])
    
    return train_ds, val_ds, test_ds

# ==============================================================================
# 2. Visualization (Task 1)
# ==============================================================================
def denormalize(tensor):
    """Reverts normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def visualize_samples(dataset, num_samples=3):
    """
    Displays samples from the dataset to visualize the image and binary mask.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        # Get random sample
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]
        
        # Denormalize image
        img_vis = denormalize(image).permute(1, 2, 0).numpy()
        mask_vis = mask.squeeze().numpy()
        
        # Check unique mask values (Task 1 requirement)
        unique_vals = np.unique(mask_vis)
        print(f"Sample {i+1} - Unique mask values: {unique_vals}")
        
        axes[i][0].imshow(img_vis)
        axes[i][0].set_title(f"Image {idx}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(f"Binary Mask (1=Animal, 0=Bg)")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig('task1_visualization.png')
    print("Saved visualization to 'task1_visualization.png'")
    plt.show()

if __name__ == "__main__":
    print("Task 1, 2, 3: Dataset Preparation and Visualization")
    
    # 1. Get datasets (Downloads if not exists)
    train_set, val_set, test_set = get_datasets(root_dir='./data')
    
    print(f"Dataset sizes:")
    print(f" - Train: {len(train_set)}")
    print(f" - Validation: {len(val_set)}")
    print(f" - Test: {len(test_set)}")
    
    # 2. Visualize some training samples (will include augmentations)
    print("\nVisualizing Training Samples (with augmentations)...")
    visualize_samples(train_set, num_samples=3)
    
    # 3. Create DataLoaders for future use
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    
    print("\nDataLoaders are ready for Phase 2!")
