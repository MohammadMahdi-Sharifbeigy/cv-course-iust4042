import cv2
import matplotlib.pyplot as plt
import os
from dataset_preparation import VehicleVOCDataset
try:
    from ultralytics import YOLO
except ImportError:
    pass

def find_busiest_yolo_image(dataset_root='./data', yolo_images_dir='./yolo_dataset/images/test'):
    """Return the converted test image with the highest number of GT boxes."""
    dataset = VehicleVOCDataset(dataset_root, image_set='test', download=False)
    best_dataset_idx = 0
    best_box_count = -1

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        box_count = len(target['boxes'])
        if box_count > best_box_count:
            best_dataset_idx = idx
            best_box_count = box_count

    real_idx = dataset.indices[best_dataset_idx]
    image_path = os.path.join(yolo_images_dir, f"{real_idx:06d}.jpg")
    return image_path, best_box_count

# ==============================================================================
# Question 2 - Section 5 - Experiment B: NMS Tuning (Task 9)
# ==============================================================================

def test_nms_thresholds(image_path=None, dataset_root='./data',
                        yolo_images_dir='./yolo_dataset/images/test',
                        save_path='task9_nms_tuning.png'):
    """
    Task 9: Runs inference on a crowded image using 3 different NMS IoU thresholds.
    If image_path is None, auto-selects the validation/test image with the
    largest number of ground-truth vehicle boxes.
    """
    print("--- Experiment B: NMS Tuning ---")
    
    # We use the pre-trained nano model if fine-tuned is not available just to demonstrate NMS
    model_path = 'yolo_runs/yolov8n_vehicles/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Fine-tuned model {model_path} not found. Falling back to default yolov8n.pt for NMS demonstration.")
        model_path = 'yolov8n.pt'
        
    model = YOLO(model_path)
    
    if image_path is None:
        try:
            image_path, box_count = find_busiest_yolo_image(dataset_root, yolo_images_dir)
            print(f"Selected busiest test image: {image_path} ({box_count} GT boxes)")
        except Exception as exc:
            print(f"Could not auto-select busiest image: {exc}")
            image_path = None

    if image_path is None or not os.path.exists(image_path):
        # Pick the first image from test set as fallback
        test_dir = yolo_images_dir
        if os.path.exists(test_dir) and os.listdir(test_dir):
            image_path = os.path.join(test_dir, sorted(os.listdir(test_dir))[0])
        else:
            print("No test images found. Please run dataset_preparation.py first.")
            return None

    # NMS IoU Thresholds to test
    nms_thresholds = [0.1, 0.5, 0.9]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, iou_thresh in enumerate(nms_thresholds):
        print(f"Running inference with NMS IoU = {iou_thresh}...")
        
        # YOLO inference with specific IoU threshold
        # conf is set relatively low to ensure we get many overlapping boxes to suppress
        results = model.predict(image_path, iou=iou_thresh, conf=0.1, verbose=False)
        
        # Plotting the annotated image
        annotated_img = results[0].plot()
        
        # Convert BGR to RGB for matplotlib
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(annotated_img)
        axes[i].set_title(f"NMS IoU Threshold: {iou_thresh}\nBoxes Detected: {len(results[0].boxes)}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved NMS tuning comparison to '{save_path}'")
    return save_path, image_path

if __name__ == "__main__":
    test_nms_thresholds()
