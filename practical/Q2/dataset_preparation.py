import os
import shutil
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import yaml
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# Question 2 - Section 1: Dataset Creation & Preparation
# ==============================================================================

# Target classes and their mappings
TARGET_CLASSES = ['car', 'bus', 'bicycle', 'motorbike']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

# ==============================================================================
# Task 1 & 2: PyTorch/Torchvision Format (Filter and Dataset Wrapper)
# ==============================================================================

class VehicleVOCDataset(Dataset):
    """
    Task 2: Custom Dataset wrapper for PyTorch/Torchvision format.
    Filters PASCAL VOC 2007 to only include 'Vehicles Only'.
    Returns targets as a dictionary with 'boxes' (FloatTensor) and 'labels' (Int64Tensor).
    """
    def __init__(self, root_dir, image_set='trainval', download=True, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.voc = VOCDetection(root=root_dir, year='2007', image_set=image_set, download=download)
        
        # Task 1: Filter Dataset
        self.indices = []
        self.class_counts = {cls: 0 for cls in TARGET_CLASSES}
        self._filter_dataset()

    def get_class_distribution(self):
        """Task 1 sanity check: return {class_name: instance_count} for plotting."""
        return dict(self.class_counts)

    def class_distribution(self):
        """Alias used by the Q2 pipeline notebook."""
        return self.get_class_distribution()

    def _filter_dataset(self):
        print(f"Filtering {self.voc.image_set} dataset for vehicles...")
        class_counts = self.class_counts

        for idx in tqdm(range(len(self.voc))):
            _, target = self.voc[idx]
            annotations = target['annotation']['object']
            if not isinstance(annotations, list):
                annotations = [annotations]

            has_vehicle = False
            for obj in annotations:
                obj_name = obj['name'].lower()
                if obj_name in CLASS_TO_ID:
                    has_vehicle = True
                    class_counts[obj_name] += 1

            if has_vehicle:
                self.indices.append(idx)

        print(f"\nFiltered {len(self.indices)} images.")
        print(f"Class Instances Distribution: {class_counts}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.voc[real_idx]
        
        annotations = target['annotation']['object']
        if not isinstance(annotations, list):
            annotations = [annotations]
            
        boxes = []
        labels = []
        
        # Task 2 parsing
        for obj in annotations:
            obj_name = obj['name'].lower()
            if obj_name in CLASS_TO_ID:
                # Extract bbox (Pascal VOC is 1-indexed, but usually we just read it)
                bndbox = obj['bndbox']
                xmin = float(bndbox['xmin'])
                ymin = float(bndbox['ymin'])
                xmax = float(bndbox['xmax'])
                ymax = float(bndbox['ymax'])
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS_TO_ID[obj_name])
                
        boxes = torch.tensor(boxes, dtype=torch.float32) # FloatTensor
        labels = torch.tensor(labels, dtype=torch.int64) # Int64Tensor
        
        target_dict = {
            "boxes": boxes,
            "labels": labels
        }
        
        if self.transforms:
            img, target_dict = self.transforms(img, target_dict)
            
        return img, target_dict


# ==============================================================================
# Task 3: Ultralytics/YOLO Format Conversion
# ==============================================================================

def convert_to_yolo_format(dataset, output_dir, split_name):
    """
    Converts the filtered dataset into YOLO txt format.
    Format: [class_id x_center y_center width height] (normalized 0 to 1)
    """
    images_dir = os.path.join(output_dir, 'images', split_name)
    labels_dir = os.path.join(output_dir, 'labels', split_name)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Converting {split_name} split to YOLO format in {output_dir}...")
    
    for idx in tqdm(range(len(dataset))):
        img, target_dict = dataset[idx]
        width, height = img.size # PIL Image size
        
        # Original voc index (for unique naming)
        real_idx = dataset.indices[idx]
        img_filename = f"{real_idx:06d}.jpg"
        label_filename = f"{real_idx:06d}.txt"
        
        # Save image
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        # Save labels
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            boxes = target_dict['boxes']
            labels = target_dict['labels']
            
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i].tolist()
                class_id = labels[i].item()
                
                # YOLO format conversion (normalized)
                x_center = ((xmin + xmax) / 2.0) / width
                y_center = ((ymin + ymax) / 2.0) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                # Clamp between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                box_width = max(0.0, min(1.0, box_width))
                box_height = max(0.0, min(1.0, box_height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

def create_yaml(output_dir):
    """Creates the data.yaml file required by YOLO."""
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/trainval',
        'val': 'images/test', # Using test as val since we don't have a separate split
        'names': {v: k for k, v in CLASS_TO_ID.items()}
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Created YOLO config at {yaml_path}")


if __name__ == "__main__":
    data_root = './data'
    yolo_output_dir = './yolo_dataset'
    
    print("--- Task 1 & 2: PyTorch Dataset Wrapper ---")
    # Load and filter Train/Val
    trainval_dataset = VehicleVOCDataset(data_root, image_set='trainval', download=True)
    # Load and filter Test (for evaluation)
    test_dataset = VehicleVOCDataset(data_root, image_set='test', download=True)
    
    print("\nTesting PyTorch output format for 1 sample:")
    img, target = trainval_dataset[0]
    print(f"Image Type: {type(img)}, Size: {img.size}")
    print(f"Target 'boxes' type: {type(target['boxes'])}, dtype: {target['boxes'].dtype}")
    print(f"Target 'labels' type: {type(target['labels'])}, dtype: {target['labels'].dtype}")
    
    print("\n--- Task 3: YOLO Format Conversion ---")
    convert_to_yolo_format(trainval_dataset, yolo_output_dir, 'trainval')
    convert_to_yolo_format(test_dataset, yolo_output_dir, 'test')
    create_yaml(yolo_output_dir)
    print("\nDataset preparation completely finished!")
