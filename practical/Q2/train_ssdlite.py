import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from dataset_preparation import VehicleVOCDataset
import functools
from tqdm import tqdm

# ==============================================================================
# Question 2 - Section 3: Fine-Tuning SSDLite (Torchvision)
# ==============================================================================

def get_ssdlite_model(num_classes=5): # 4 vehicles + 1 background
    """
    Task 5: Architecture Modification
    """
    print("Loading pre-trained SSDLite320 MobileNet V3 Large...")
    # 1. Load the pre-trained SSDLite model
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    
    # 2. Freeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone weights frozen.")
        
    # 3. Replace the classification head
    # We need to extract the in_channels from the existing classification head
    # The classification_head has a module_list where each item is a Sequential or Conv block
    in_channels = []
    for m in model.head.classification_head.module_list:
        # Get the first conv layer in the sequence to find its input channels
        if isinstance(m, torch.nn.Sequential):
            in_channels.append(m[0][0].in_channels)
        else:
            in_channels.append(m.in_channels)
            
    num_anchors = model.anchor_generator.num_anchors_per_location()

    norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    
    print(f"Replacing classification head for {num_classes} classes...")
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
    
    return model

def collate_fn(batch):
    """Custom collate function to handle dictionary targets."""
    return tuple(zip(*batch))

def prepare_ssdlite_targets(targets, device):
    """
    Move targets to device and shift VOC vehicle labels from YOLO-style 0..3 to
    Torchvision detection labels 1..4, where 0 is reserved for background.
    """
    prepared = []
    for target in targets:
        target_on_device = {k: v.to(device) for k, v in target.items()}
        target_on_device["labels"] = target_on_device["labels"] + 1
        prepared.append(target_on_device)
    return prepared

@torch.no_grad()
def evaluate_ssdlite_loss(model, loader, device):
    """Compute validation loss while preserving the caller's model mode."""
    was_training = model.training
    model.train()
    val_loss = 0.0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = prepare_ssdlite_targets(targets, device)
        loss_dict = model(images, targets)
        val_loss += sum(loss for loss in loss_dict.values()).item()

    if not was_training:
        model.eval()

    return val_loss / max(len(loader), 1)

def train_ssdlite(num_epochs=50, batch_size=8, device='cuda', lr=1e-4,
                  data_root='./data', train_split='trainval', val_split='test',
                  save_path='ssdlite_finetuned.pth', use_amp=True,
                  scheduler_step_size=None, scheduler_gamma=0.1):
    """
    Task 6: The Forward Pass

    Returns:
        (model, history) where history contains train_loss, val_loss, and lr.
    """
    print(f"Starting SSDLite training on {device}...")
    
    # Transformations for Object Detection: Convert image to tensor (0-1)
    # PyTorch's SSD model expects images in [0, 1] range as standard FloatTensors
    from torchvision.transforms import functional as F
    def transform(img, target):
        img = F.to_tensor(img)
        return img, target
        
    # 1. Load Dataset
    train_dataset = VehicleVOCDataset(data_root, image_set=train_split, download=False, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = None
    if val_split:
        val_dataset = VehicleVOCDataset(data_root, image_set=val_split, download=False, transforms=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_fn)
    
    # 2. Get Model
    model = get_ssdlite_model(num_classes=5).to(device)
    
    # Only optimize the parameters that require gradients (the head)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = None
    if scheduler_step_size:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size,
                                              gamma=scheduler_gamma)
    amp_enabled = bool(use_amp and device == 'cuda' and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    model.train() # Set to training mode (computes losses)
    
    # 3. Training Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in pbar:
            images = [image.to(device) for image in images]
            targets = prepare_ssdlite_targets(targets, device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # In train mode, SSDLite returns bbox_regression and classification losses.
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        avg_val_loss = None
        if val_loader is not None:
            avg_val_loss = evaluate_ssdlite_loss(model, val_loader, device)
            history["val_loss"].append(avg_val_loss)

        if scheduler is not None:
            scheduler.step()

        if avg_val_loss is None:
            print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned SSDLite model to '{save_path}'")
    return model, history

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # We run 1 epoch just to test the implementation for the assignment.
    # The prompt asks for 50 epochs, but this script can be executed fully by the user.
    train_ssdlite(num_epochs=1, batch_size=4, device=device)
