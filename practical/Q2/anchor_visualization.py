import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.image_list import ImageList
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ==============================================================================
# Question 2 - Section 2: Anchor Visualization (Priors)
# ==============================================================================

def extract_anchors():
    """
    Task 3: Extracts the anchors from SSDLite by running a dummy tensor.
    """
    print("Loading SSDLite320 MobileNet V3 Large model...")
    # Load model with pre-trained weights
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()

    # Create dummy tensor 3x320x320
    print("Creating dummy tensor and extracting anchors...")
    dummy_tensor = torch.zeros(1, 3, 320, 320)
    
    # In torchvision SSD, we first extract feature maps from the backbone
    with torch.no_grad():
        features = model.backbone(dummy_tensor)
        
        # Then pass them to the anchor generator
        # Anchor generator requires an ImageList and a list of feature maps
        image_list = ImageList(dummy_tensor, [(320, 320)])
        feature_maps = list(features.values())
        
        anchors = model.anchor_generator(image_list, feature_maps)
    
    # anchors is a list of tensors (one per image). We take the first one.
    # It contains anchors for ALL feature maps concatenated together.
    # Shape of anchors[0] is [N, 4] where N is total number of anchors.
    # Format is [xmin, ymin, xmax, ymax]
    
    return anchors[0], feature_maps, model.anchor_generator

def visualize_center_anchors(anchors, image_size=(320, 320), background_image=None,
                             save_path='task4_anchor_visualization.png'):
    """
    Task 4: Visualizes the anchors corresponding to the center of the image.

    background_image: optional PIL.Image (a real PASCAL VOC sample). It is
    resized to image_size so the priors are overlaid on actual content rather
    than a blank canvas. If None, falls back to a light-gray background.
    """
    print("\nVisualizing central anchors...")
    
    # We want to find anchors whose centers are close to the center of the image (160, 160)
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    
    # Calculate centers of all anchors
    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2.0
    
    # Calculate distance to image center
    distances = torch.sqrt((anchor_centers_x - center_x)**2 + (anchor_centers_y - center_y)**2)
    
    # Find the minimum distance (this gives us the closest grid cell to the center)
    min_dist = distances.min()
    
    # Select all anchors that share this minimum distance (the ones belonging to the center cell)
    # Using a small tolerance for floating point comparisons
    center_anchor_indices = torch.where(distances <= min_dist + 1e-4)[0]
    center_anchors = anchors[center_anchor_indices]
    
    print(f"Found {len(center_anchors)} anchors at the center of the image.")
    
    # Plotting
    fig, ax = plt.subplots(1, figsize=(8, 8))
    
    # Use a real PASCAL VOC sample if provided, else a light-gray canvas.
    if background_image is not None:
        bg = background_image.convert('RGB').resize(image_size)
        ax.imshow(np.asarray(bg))
    else:
        dummy_img = np.ones((image_size[1], image_size[0], 3)) * 0.9
        ax.imshow(dummy_img)

    # Non-deprecated colormap access (matplotlib >= 3.7).
    cmap = plt.get_cmap('hsv')
    n = max(len(center_anchors), 1)
    colors = lambda i: cmap(i / n)
    
    for i, anchor in enumerate(center_anchors):
        xmin, ymin, xmax, ymax = anchor.tolist()
        width = xmax - xmin
        height = ymax - ymin
        
        rect = patches.Rectangle((xmin, ymin), width, height, 
                                 linewidth=2, edgecolor=colors(i), facecolor='none', 
                                 label=f'AR: {width/height:.2f}')
        ax.add_patch(rect)
    
    # Mark the center
    ax.plot(center_x, center_y, 'ko', markersize=5, label='Grid Center')
    
    plt.title('SSDLite Anchors at the Center Grid Cell')
    plt.xlim(0, image_size[0])
    plt.ylim(image_size[1], 0) # Invert Y axis to match image coordinates
    plt.legend(loc='upper right')
    
    plt.savefig(save_path)
    print(f"Saved anchor visualization to {save_path}")
    return center_anchors
    
if __name__ == "__main__":
    all_anchors, fmaps, generator = extract_anchors()
    print(f"Total anchors generated: {all_anchors.shape[0]}")
    visualize_center_anchors(all_anchors)
