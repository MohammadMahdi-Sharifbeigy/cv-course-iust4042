import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_datasets, DataLoader, denormalize
from unet import UNet
from utils import calculate_metrics

# ==============================================================================
# Task 8, 9, 10: Section 3 - U-Net Evaluation
# ==============================================================================

def evaluate_model(model_path='best_unet_model.pth', device='cuda'):
    print(f"Loading datasets...")
    _, _, test_set = get_datasets(root_dir='./data')
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Loading model from {model_path}...")
    model = UNet(n_channels=3, n_classes=1).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first by running train.py")
        return model, test_set
        
    model.eval()
    
    # Task 8: Quantitative Evaluation
    print("\n--- Task 8: Quantitative Evaluation ---")
    total_iou = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            iou, dice = calculate_metrics(outputs, masks)
            
            total_iou += iou * images.size(0)
            total_dice += dice * images.size(0)
            
    mean_iou = total_iou / len(test_loader.dataset)
    mean_dice = total_dice / len(test_loader.dataset)
    
    print(f"Test Set Evaluation => Mean IoU: {mean_iou:.4f} | Mean Dice Score: {mean_dice:.4f}")
    
    return model, test_set

def qualitative_evaluation(model, dataset, device='cuda', num_samples=3):
    """
    Task 9: Qualitative Evaluation (Display Original, GT, Prediction, and Overlay)
    """
    print("\n--- Task 9: Qualitative Evaluation ---")
    model.eval()
    
    # Pick random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Predict
            img_tensor = image.unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()
            
            # Prepare for visualization
            img_vis = denormalize(image).permute(1, 2, 0).numpy()
            gt_mask = mask.squeeze().numpy()
            
            # Subplot setup
            ax_orig = axes[i][0] if num_samples > 1 else axes[0]
            ax_gt = axes[i][1] if num_samples > 1 else axes[1]
            ax_pred = axes[i][2] if num_samples > 1 else axes[2]
            ax_overlay = axes[i][3] if num_samples > 1 else axes[3]
            
            # 1. Original Image
            ax_orig.imshow(img_vis)
            ax_orig.set_title("Original Image")
            ax_orig.axis('off')
            
            # 2. Ground Truth Mask
            ax_gt.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            ax_gt.set_title("Ground Truth Mask")
            ax_gt.axis('off')
            
            # 3. Predicted Mask
            ax_pred.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis('off')
            
            # 4. Overlay
            # Create a red mask overlay for the prediction
            overlay = img_vis.copy()
            # Highlight predicted animal pixels in red
            overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
            
            ax_overlay.imshow(overlay)
            ax_overlay.set_title("Overlay (Pred in Red)")
            ax_overlay.axis('off')
            
    plt.tight_layout()
    plt.savefig('task9_qualitative_eval.png')
    print("Saved qualitative results to 'task9_qualitative_eval.png'")

def error_analysis(model, dataset, device='cuda', num_samples=2):
    """
    Task 10: Error Analysis - Finds samples with low IoU
    """
    print("\n--- Task 10: Error Analysis ---")
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    worst_samples = []
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            if len(worst_samples) >= num_samples:
                break
                
            img_tensor = image.to(device)
            mask_tensor = mask.to(device)
            
            output = model(img_tensor)
            iou, dice = calculate_metrics(output, mask_tensor)
            
            # If IoU is particularly low, it's a good candidate for error analysis
            if iou < 0.5:
                worst_samples.append((image.squeeze(), mask.squeeze(), output.squeeze(), iou.item() if isinstance(iou, torch.Tensor) else iou, dice.item() if isinstance(dice, torch.Tensor) else dice))
                
    if not worst_samples:
        print("Could not find severe error samples easily. Model might be performing very well!")
        return
        
    fig, axes = plt.subplots(len(worst_samples), 4, figsize=(16, 4 * len(worst_samples)))
    
    for i, (image, gt_mask, output, iou, dice) in enumerate(worst_samples):
        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
        img_vis = denormalize(image).permute(1, 2, 0).numpy()
        gt_mask = gt_mask.numpy()
        
        # Determine likely cause of error
        print(f"Error Sample {i+1} saved to plot. IoU: {iou:.4f}, Dice: {dice:.4f}")
        
        ax_orig = axes[i][0] if len(worst_samples) > 1 else axes[0]
        ax_gt = axes[i][1] if len(worst_samples) > 1 else axes[1]
        ax_pred = axes[i][2] if len(worst_samples) > 1 else axes[2]
        ax_overlay = axes[i][3] if len(worst_samples) > 1 else axes[3]
        
        ax_orig.imshow(img_vis)
        ax_orig.set_title("Original Image (Error Case)")
        ax_orig.axis('off')
        
        ax_gt.imshow(gt_mask, cmap='gray')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis('off')
        
        ax_pred.imshow(pred_mask, cmap='gray')
        ax_pred.set_title("Failed Prediction")
        ax_pred.axis('off')
        
        overlay = img_vis.copy()
        overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay (Error Analysis)")
        ax_overlay.axis('off')
        
    plt.tight_layout()
    plt.savefig('task10_error_analysis.png')
    print("Saved error analysis results to 'task10_error_analysis.png'")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, test_dataset = evaluate_model(device=device)
    
    if os.path.exists('best_unet_model.pth'):
        qualitative_evaluation(model, test_dataset, device=device)
        error_analysis(model, test_dataset, device=device)
