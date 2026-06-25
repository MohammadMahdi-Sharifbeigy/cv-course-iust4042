import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from dataset import get_datasets, denormalize, DataLoader
from utils import calculate_metrics

# Try to import Segment Anything Model (SAM)
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("WARNING: Segment Anything Model not found!")
    print("Please install it using:")
    print("pip install git+https://github.com/facebookresearch/segment-anything.git")

# ==============================================================================
# Task 11, 12, 13, 14: Section 4 - Segment Anything Model (Zero-Shot)
# ==============================================================================

def download_sam_weights():
    """Downloads the ViT-B weights for SAM if they don't exist."""
    checkpoint_path = "sam_vit_b_01ec64.pth"
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM ViT-B checkpoint (approx 358MB)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, checkpoint_path)
        print("Download complete.")
    return checkpoint_path

def evaluate_sam_zero_shot(num_samples=5, device='cuda'):
    if not SAM_AVAILABLE:
        print("SAM is not installed. Please install it to run this cell.")
        return
        
    print("\n--- Task 11: Loading SAM Model ---")
    checkpoint = download_sam_weights()
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to(device=device)
    
    # Task 12: Initialize Automatic Mask Generator
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print("Loading test dataset...")
    _, _, test_set = get_datasets(root_dir='./data')
    # Use batch size 1 since SAM processes one image at a time
    loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    total_best_iou = 0.0
    total_best_dice = 0.0
    
    print(f"\n--- Task 12 & 13: Generating Masks and Finding Best Match ---")
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    samples_processed = 0
    for image, gt_mask in loader:
        if samples_processed >= num_samples:
            break
            
        # Denormalize image and convert to HWC uint8 format for SAM
        img_np = denormalize(image.squeeze()).permute(1, 2, 0).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # Ground Truth Mask
        gt_mask_np = gt_mask.squeeze().numpy()
        gt_mask_tensor = gt_mask.to(device)
        
        # Generate Masks using SAM
        print(f"Processing image {samples_processed + 1}/{num_samples}...")
        sam_result = mask_generator.generate(img_uint8)
        
        if len(sam_result) == 0:
            continue
            
        # Find the best mask (Highest IoU with Ground Truth)
        best_iou = 0.0
        best_dice = 0.0
        best_mask = None
        
        for ann in sam_result:
            # SAM outputs boolean masks
            sam_mask = ann['segmentation'].astype(np.float32)
            sam_mask_tensor = torch.tensor(sam_mask).unsqueeze(0).unsqueeze(0).to(device)
            
            # Note: We pass raw boolean as probabilities here for metric calc
            # We don't need sigmoid since it's already 0 or 1
            iou, dice = calculate_metrics((sam_mask_tensor - 0.5) * 10, gt_mask_tensor) # Simulate logits
            
            if iou > best_iou:
                best_iou = iou
                best_dice = dice
                best_mask = sam_mask
                
        total_best_iou += best_iou
        total_best_dice += best_dice
        
        # Visualization
        ax_orig = axes[samples_processed][0] if num_samples > 1 else axes[0]
        ax_gt = axes[samples_processed][1] if num_samples > 1 else axes[1]
        ax_sam_all = axes[samples_processed][2] if num_samples > 1 else axes[2]
        ax_best = axes[samples_processed][3] if num_samples > 1 else axes[3]
        
        # Original
        ax_orig.imshow(img_uint8)
        ax_orig.set_title("Original Image")
        ax_orig.axis('off')
        
        # Ground Truth
        ax_gt.imshow(gt_mask_np, cmap='gray')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis('off')
        
        # All SAM Masks overlaid
        ax_sam_all.imshow(img_uint8)
        # Show all masks in random colors
        for ann in sam_result:
            m = ann['segmentation']
            color_mask = np.random.random(3)
            ax_sam_all.imshow(np.dstack((m, m, m)) * color_mask, alpha=0.3)
        ax_sam_all.set_title(f"All SAM Masks ({len(sam_result)})")
        ax_sam_all.axis('off')
        
        # Best Mask
        ax_best.imshow(best_mask, cmap='gray')
        ax_best.set_title(f"Best SAM Mask (IoU: {best_iou:.2f})")
        ax_best.axis('off')
        
        samples_processed += 1
        
    mean_iou = total_best_iou / num_samples
    mean_dice = total_best_dice / num_samples
    
    print("\n--- Task 13: SAM Quantitative Results ---")
    print(f"Mean IoU of Best SAM Masks: {mean_iou:.4f}")
    print(f"Mean Dice of Best SAM Masks: {mean_dice:.4f}")
    
    plt.tight_layout()
    plt.savefig('task13_sam_evaluation.png')
    print("Saved SAM evaluation visualization to 'task13_sam_evaluation.png'")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_sam_zero_shot(num_samples=5, device=device)
