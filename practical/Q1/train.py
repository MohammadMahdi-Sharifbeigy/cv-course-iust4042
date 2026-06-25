import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import get_datasets, DataLoader
from unet import UNet
from utils import BCEDiceLoss, calculate_metrics

# ==============================================================================
# Task 6 & 7: Training Loop and Analysis
# ==============================================================================

def train_model(num_epochs=10, batch_size=8, learning_rate=1e-3, device='cuda'):
    print(f"Starting training on device: {device}")
    
    # 1. Prepare Data
    # Assuming datasets were downloaded via dataset.py
    train_set, val_set, test_set = get_datasets(root_dir='./data')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize Model, Loss, Optimizer
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize TensorBoard Writer
    os.makedirs('lightning_logs', exist_ok=True)
    writer = SummaryWriter(log_dir='lightning_logs')
    global_step = 0
    
    # Metrics tracking (for Task 7)
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    # 3. Training Loop (Task 6)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training pass
        print(f"Epoch {epoch+1}/{num_epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
            # Log step-level metric
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
            # Save step-level checkpoint (overwriting latest to save space)
            torch.save(model.state_dict(), 'lightning_logs/latest_step_checkpoint.pth')
            
            global_step += 1
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # Calculate metrics
                iou, dice = calculate_metrics(outputs, masks)
                val_iou += iou * images.size(0)
                val_dice += dice * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_iou = val_iou / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}\n")
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Metrics/val_iou', val_iou, epoch)
        writer.add_scalar('Metrics/val_dice', val_dice, epoch)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        
        # Save best model
        if epoch == 0 or val_iou > max(history['val_iou'][:-1] + [0]):
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print("=> Saved best model")

    # Close TensorBoard Writer
    writer.close()

    # 4. Analysis and Plotting (Task 7)
    print("Training complete! Generating analysis plots...")
    plot_training_history(history, num_epochs)
    
    return model, history

def plot_training_history(history, epochs):
    """
    Plots Training Loss, Validation Loss, and Validation IoU (Task 7)
    """
    x = range(1, epochs + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(x, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(x, history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE + Dice)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Metrics
    plt.subplot(1, 2, 2)
    plt.plot(x, history['val_iou'], label='Validation IoU', color='green', marker='o')
    plt.plot(x, history['val_dice'], label='Validation Dice', color='orange', marker='s')
    plt.title('Validation Metrics (IoU & Dice)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('task7_training_analysis.png')
    print("Saved training analysis plot to 'task7_training_analysis.png'")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For a quick homework test, 5-10 epochs are usually enough. We set 10 by default.
    model, history = train_model(num_epochs=10, batch_size=16, learning_rate=1e-3, device=device)
