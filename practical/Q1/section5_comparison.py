import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_datasets, denormalize, DataLoader
from unet import UNet
from utils import calculate_metrics

# Try importing SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM is not installed. Qualitative and Quantitative comparison will fail.")

# ==============================================================================
# Task: Section 5 - Experiment Design and Model Comparison
# ==============================================================================

def compare_models(num_qualitative_samples=3, num_quantitative_samples=50, device='cuda'):
    print(f"Loading test dataset...")
    _, _, test_set = get_datasets(root_dir='./data')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    print("\nLoading U-Net Model...")
    unet_model = UNet(n_channels=3, n_classes=1).to(device)
    unet_loaded = False
    try:
        unet_model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
        unet_model.eval()
        unet_loaded = True
        print("U-Net loaded successfully.")
    except FileNotFoundError:
        print("WARNING: 'best_unet_model.pth' not found. U-Net predictions will be random!")

    if SAM_AVAILABLE:
        print("\nLoading SAM Model (Zero-Shot)...")
        checkpoint = "sam_vit_b_01ec64.pth"
        if not os.path.exists(checkpoint):
            print("SAM weights not found. Please run sam_evaluation.py first to download them.")
            return
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        return

    # --- Experiment A: Qualitative Comparison ---
    print("\n=== Experiment A: Qualitative Comparison ===")
    fig, axes = plt.subplots(num_qualitative_samples, 4, figsize=(16, 4 * num_qualitative_samples))
    
    samples_processed = 0
    unet_total_iou = 0.0
    unet_total_dice = 0.0
    sam_total_iou = 0.0
    sam_total_dice = 0.0
    
    print("\nRunning Evaluation (Qualitative and Quantitative)...")
    print(f"Evaluating {num_quantitative_samples} samples")
    
    with torch.no_grad():
        for image, gt_mask in test_loader:
            if samples_processed >= num_quantitative_samples:
                break
            
            # Prepare image & mask
            img_tensor = image.to(device)
            gt_mask_tensor = gt_mask.to(device)
            gt_mask_np = gt_mask.squeeze().numpy()
            
            # Denormalize for SAM and Plotting
            img_np = denormalize(image.squeeze()).permute(1, 2, 0).numpy()
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            # --- U-Net Prediction ---
            unet_out = unet_model(img_tensor)
            unet_iou, unet_dice = calculate_metrics(unet_out, gt_mask_tensor)
            unet_mask_np = (torch.sigmoid(unet_out) > 0.5).float().cpu().squeeze().numpy()
            
            unet_total_iou += unet_iou
            unet_total_dice += unet_dice
            
            # --- SAM Prediction ---
            sam_result = mask_generator.generate(img_uint8)
            
            best_sam_iou = 0.0
            best_sam_dice = 0.0
            best_sam_mask = np.zeros_like(gt_mask_np)
            
            if len(sam_result) > 0:
                for ann in sam_result:
                    sam_mask = ann['segmentation'].astype(np.float32)
                    sam_mask_tensor = torch.tensor(sam_mask).unsqueeze(0).unsqueeze(0).to(device)
                    # Use metric function
                    iou, dice = calculate_metrics((sam_mask_tensor - 0.5) * 10, gt_mask_tensor)
                    if iou > best_sam_iou:
                        best_sam_iou = iou
                        best_sam_dice = dice
                        best_sam_mask = sam_mask
                        
            sam_total_iou += best_sam_iou
            sam_total_dice += best_sam_dice
            
            # --- Plotting Qualitative (Only first few samples) ---
            if samples_processed < num_qualitative_samples:
                ax_orig = axes[samples_processed][0] if num_qualitative_samples > 1 else axes[0]
                ax_gt = axes[samples_processed][1] if num_qualitative_samples > 1 else axes[1]
                ax_unet = axes[samples_processed][2] if num_qualitative_samples > 1 else axes[2]
                ax_sam = axes[samples_processed][3] if num_qualitative_samples > 1 else axes[3]
                
                ax_orig.imshow(img_uint8)
                ax_orig.set_title("Original Image")
                ax_orig.axis('off')
                
                ax_gt.imshow(gt_mask_np, cmap='gray')
                ax_gt.set_title("Ground Truth")
                ax_gt.axis('off')
                
                ax_unet.imshow(unet_mask_np, cmap='gray')
                ax_unet.set_title(f"U-Net (IoU: {unet_iou:.2f})")
                ax_unet.axis('off')
                
                ax_sam.imshow(best_sam_mask, cmap='gray')
                ax_sam.set_title(f"SAM Best (IoU: {best_sam_iou:.2f})")
                ax_sam.axis('off')
                
            samples_processed += 1
            print(f"Processed {samples_processed}/{num_quantitative_samples}...", end='\r')
            
    print("\n\nQualitative plot generated!")
    plt.tight_layout()
    plt.savefig('task15_model_comparison_qualitative.png')
    print("Saved qualitative comparison to 'task15_model_comparison_qualitative.png'")
    
    # --- Experiment B: Quantitative Comparison ---
    print("\n=== Experiment B: Quantitative Comparison ===")
    mean_unet_iou = unet_total_iou / samples_processed
    mean_unet_dice = unet_total_dice / samples_processed
    mean_sam_iou = sam_total_iou / samples_processed
    mean_sam_dice = sam_total_dice / samples_processed
    
    print("\n" + "="*50)
    print(f"{'Model':<15} | {'Mean IoU':<12} | {'Mean Dice Score':<15}")
    print("-" * 50)
    print(f"{'U-Net':<15} | {mean_unet_iou:<12.4f} | {mean_unet_dice:<15.4f}")
    print(f"{'SAM (Zero-Shot)':<15} | {mean_sam_iou:<12.4f} | {mean_sam_dice:<15.4f}")
    print("="*50)
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compare_models(num_qualitative_samples=3, num_quantitative_samples=50, device=device)
