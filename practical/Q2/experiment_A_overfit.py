import os
import shutil
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from dataset_preparation import VehicleVOCDataset
from train_ssdlite import get_ssdlite_model, collate_fn, prepare_ssdlite_targets
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO

# ==============================================================================
# Question 2 - Section 5 - Experiment A: The Overfit Test (Task 8)
# ==============================================================================

def create_yolo_micro_dataset(dataset, indices, output_dir='./yolo_micro'):
    """Creates a micro-dataset of 10 images for YOLO training."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    
    for count, idx in enumerate(indices):
        img, target = dataset[idx]
        width, height = img.size
        
        img_path = os.path.join(output_dir, 'images', 'train', f"{count:03d}.jpg")
        img.save(img_path)
        
        label_path = os.path.join(output_dir, 'labels', 'train', f"{count:03d}.txt")
        with open(label_path, 'w') as f:
            for i in range(len(target['boxes'])):
                xmin, ymin, xmax, ymax = target['boxes'][i].tolist()
                cid = target['labels'][i].item()
                xc = max(0, min(1, ((xmin + xmax) / 2.0) / width))
                yc = max(0, min(1, ((ymin + ymax) / 2.0) / height))
                w = max(0, min(1, (xmax - xmin) / width))
                h = max(0, min(1, (ymax - ymin) / height))
                f.write(f"{cid} {xc} {yc} {w} {h}\n")
                
    import yaml
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/train', # Validate on the same set
        'names': {0: 'car', 1: 'bus', 2: 'bicycle', 3: 'motorbike'}
    }
    with open(os.path.join(output_dir, 'micro_data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
        
    return os.path.join(output_dir, 'micro_data.yaml')

def train_and_visualize_overfit(epochs=50, device='cuda'):
    print("--- Experiment A: The Overfit Test ---")
    
    # 1. Create Micro Dataset (10 images)
    from torchvision.transforms import functional as F
    def transform(img, target): return F.to_tensor(img), target
    
    full_dataset = VehicleVOCDataset('./data', image_set='trainval', download=False, transforms=transform)
    # Pick 10 images randomly
    indices = torch.randperm(len(full_dataset))[:10].tolist()
    micro_dataset = Subset(full_dataset, indices)
    
    # Also prepare YOLO micro dataset (which needs raw PIL images, so we use another instance)
    raw_dataset = VehicleVOCDataset('./data', image_set='trainval', download=False)
    yolo_yaml = create_yolo_micro_dataset(raw_dataset, indices)
    
    # 2. Train SSDLite
    print("\n--- Training SSDLite on 10 images ---")
    ssdlite_model = get_ssdlite_model(num_classes=5).to(device)
    optimizer = optim.Adam(ssdlite_model.parameters(), lr=1e-4)
    loader = DataLoader(micro_dataset, batch_size=2, collate_fn=collate_fn)
    
    ssdlite_losses = []
    ssdlite_model.train()
    for ep in range(epochs):
        ep_loss = 0
        for images, targets in loader:
            images = list(img.to(device) for img in images)
            targets = prepare_ssdlite_targets(targets, device)
            
            loss_dict = ssdlite_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            ep_loss += losses.item()
        ssdlite_losses.append(ep_loss / len(loader))
        if (ep+1) % 10 == 0:
            print(f"SSDLite Epoch {ep+1}/{epochs} - Loss: {ep_loss / len(loader):.4f}")
            
    # 3. Train YOLOv8n
    print("\n--- Training YOLOv8n on 10 images ---")
    yolo_model = YOLO('yolov8n.pt')
    # Use small patience and no augmentation for pure memorization
    project_dir = os.path.abspath('overfit_run')
    yolo_model.train(data=yolo_yaml, epochs=epochs, imgsz=320, project=project_dir, 
                     name='yolo_micro', device='0' if device=='cuda' else 'cpu', hsv_h=0, hsv_s=0, hsv_v=0, fliplr=0, workers=0)
    yolo_results_csv = os.path.join(project_dir, 'yolo_micro', 'results.csv')
    yolo_box_loss = None
    if os.path.exists(yolo_results_csv):
        try:
            import pandas as pd
            yolo_results = pd.read_csv(yolo_results_csv)
            train_box_cols = [c for c in yolo_results.columns if 'train/box_loss' in c]
            if train_box_cols:
                yolo_box_loss = yolo_results[train_box_cols[0]].tolist()
        except ImportError:
            print("pandas is not installed; YOLO overfit loss curve will be read in the notebook if available.")
    
    # 4. Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), ssdlite_losses, label='SSDLite Training Loss', color='blue')
    plt.title('Overfit Test: SSDLite Loss on 10 Images')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('task8_overfit_loss_ssdlite.png')
    print("Saved SSDLite loss plot to task8_overfit_loss_ssdlite.png")
    
    # 5. Visualize Memorization
    print("\nVisualizing predictions to prove memorization...")
    ssdlite_model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Original Image
            raw_img, raw_target = raw_dataset[idx]
            
            # SSDLite Predict
            tensor_img = F.to_tensor(raw_img).unsqueeze(0).to(device)
            out = ssdlite_model(tensor_img)[0]
            
            ax = axes[i]
            ax.imshow(raw_img)
            
            # Draw predicted boxes with high confidence
            for box, score, label in zip(out['boxes'], out['scores'], out['labels']):
                if score > 0.5: # Overfitted model should have high confidence
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                             linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, f"Score: {score:.2f}", color='red', fontsize=8, backgroundcolor='white')
                    
            ax.set_title(f"Train Sample {i+1}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig('task8_overfit_predictions.png')
    print("Saved memorization visualizations to task8_overfit_predictions.png")
    return {
        'ssdlite_losses': ssdlite_losses,
        'yolo_box_loss': yolo_box_loss,
        'yolo_results_csv': yolo_results_csv if os.path.exists(yolo_results_csv) else None,
        'indices': indices,
        'prediction_figure': 'task8_overfit_predictions.png',
        'loss_figure': 'task8_overfit_loss_ssdlite.png',
    }

if __name__ == "__main__":
    train_and_visualize_overfit(epochs=50)
