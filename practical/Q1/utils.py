import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Task 5: Loss Function and Evaluation Metrics
# ==============================================================================

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss
    """
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # inputs are raw logits (before sigmoid)
        # targets are binary masks (0 or 1)
        targets = targets.float()
        
        # 1. Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # 2. Dice Loss
        inputs_prob = torch.sigmoid(inputs) # Convert logits to probabilities
        
        # Flatten predictions and targets
        inputs_flat = inputs_prob.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        smooth = 1e-6
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        # 3. Combine
        combined_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return combined_loss

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculates Intersection over Union (IoU) and Dice Score for a batch.
    """
    # Convert logits to binary predictions
    preds_prob = torch.sigmoid(preds)
    preds_binary = (preds_prob > threshold).float()
    targets_binary = targets.float()
    
    # Calculate for each image in batch independently to get mean metrics
    batch_size = preds_binary.shape[0]
    iou_list = []
    dice_list = []
    
    smooth = 1e-6
    for i in range(batch_size):
        p = preds_binary[i].reshape(-1)
        t = targets_binary[i].reshape(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
        
        iou_list.append(iou.item())
        dice_list.append(dice.item())
        
    return sum(iou_list) / batch_size, sum(dice_list) / batch_size
