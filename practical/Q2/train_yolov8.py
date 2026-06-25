import os
import torch
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ultralytics is not installed. Please install it using: pip install ultralytics")

# ==============================================================================
# Question 2 - Section 4: Fine-Tuning YOLOv8n (Ultralytics)
# ==============================================================================

def train_yolo(data_yaml_path='./yolo_dataset/data.yaml', epochs=50, imgsz=320,
               device=None, model_weights='yolov8n.pt', allow_weight_download=True):
    """
    Task 7: Training Pipeline for YOLOv8n
    """
    if not ULTRALYTICS_AVAILABLE:
        print("Cannot train YOLO without the ultralytics package.")
        return
        
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found. Please run dataset_preparation.py first.")
        return
        
    if not allow_weight_download and not os.path.exists(model_weights):
        print(f"Cannot find {model_weights}. Set allow_weight_download=True to let Ultralytics fetch it.")
        return None

    print("--- Loading YOLOv8 Nano Pre-trained Model ---")
    # Load a model
    # 'yolov8n.pt' will be automatically downloaded if not present locally
    model = YOLO(model_weights)

    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f"\n--- Starting YOLOv8 Training (Epochs: {epochs}, Img Size: {imgsz}) ---")
    # Train the model
    # The Ultralytics API handles the entire training loop, loss calculation, 
    # backpropagation, evaluation, and saving best checkpoints automatically.
    results = model.train(
        data=os.path.abspath(data_yaml_path), # Path to dataset config
        epochs=epochs,                        # Number of training epochs
        imgsz=imgsz,                          # Image size to ensure fair comparison with SSDLite320
        project=os.path.abspath('yolo_runs'), # Output directory
        name='yolov8n_vehicles',              # Experiment name
        device=device,                         # Use GPU if available
        workers=0                             # Fix CUDA unknown error in Windows multiprocessing
    )
    
    print("\nTraining Complete! Check the 'yolo_runs/yolov8n_vehicles' directory for results, weights, and graphs.")
    return results

if __name__ == "__main__":
    # The prompt asks for 50 epochs
    train_yolo(epochs=50)
