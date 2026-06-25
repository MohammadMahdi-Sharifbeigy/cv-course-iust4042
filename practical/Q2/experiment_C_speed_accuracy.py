import time
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_preparation import VehicleVOCDataset
from train_ssdlite import get_ssdlite_model, collate_fn, prepare_ssdlite_targets
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

# ==============================================================================
# Question 2 - Section 5 - Experiment C: Inference Speed vs Accuracy (Task 10)
# ==============================================================================

def _load_ssdlite(device='cuda', weights_path='ssdlite_finetuned.pth'):
    model = get_ssdlite_model(num_classes=5).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded fine-tuned SSDLite weights from {weights_path}.")
    else:
        print("Fine-tuned SSDLite weights not found. Using pre-trained backbone with random vehicle head.")
    return model

def _load_yolo(model_path='yolo_runs/yolov8n_vehicles/weights/best.pt'):
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics is required for YOLO benchmarking/evaluation.")
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
    return YOLO(model_path), model_path

def benchmark_ssdlite(device='cuda', num_samples=100, weights_path='ssdlite_finetuned.pth'):
    print("--- Benchmarking SSDLite320 ---")
    model = _load_ssdlite(device=device, weights_path=weights_path)
    model.eval()
    
    from torchvision.transforms import functional as F
    def transform(img, target): return F.to_tensor(img), target
    
    dataset = VehicleVOCDataset('./data', image_set='test', download=False, transforms=transform)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn) # Batch size 1 for FPS calculation
    
    # Warmup
    dummy = [torch.randn(3, 320, 320).to(device)]
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
            
    # Benchmark FPS
    times = []
    samples_processed = 0
    with torch.no_grad():
        for images, _ in loader:
            if samples_processed >= num_samples:
                break
            
            images = list(img.to(device) for img in images)
            
            # Synchronize if using CUDA to get accurate timing
            if device == 'cuda':
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            _ = model(images)
            
            if device == 'cuda':
                torch.cuda.synchronize()
                
            end = time.perf_counter()
            times.append(end - start)
            samples_processed += 1
            
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"SSDLite FPS: {fps:.2f} (Avg time per image: {avg_time*1000:.2f} ms)")
    
    return fps

def benchmark_yolov8n(device='cuda', num_samples=100,
                      model_path='yolo_runs/yolov8n_vehicles/weights/best.pt'):
    print("\n--- Benchmarking YOLOv8n ---")
    model, resolved_model_path = _load_yolo(model_path)
    print(f"Using YOLO weights: {resolved_model_path}")
    
    dataset = VehicleVOCDataset('./data', image_set='test', download=False)
    
    # Warmup
    import cv2
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, verbose=False)
        
    times = []
    for i in range(num_samples):
        if i >= len(dataset):
            break
        img, _ = dataset[i]
        img_np = np.array(img)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.predict(img_np, verbose=False, device=0 if device == 'cuda' else 'cpu')
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"YOLOv8n FPS: {fps:.2f} (Avg time per image: {avg_time*1000:.2f} ms)")
    
    return fps

def evaluate_ssdlite_map(device='cuda', num_samples=None, weights_path='ssdlite_finetuned.pth'):
    """Evaluate fine-tuned SSDLite mAP on the VOC vehicle test split."""
    if not TORCHMETRICS_AVAILABLE:
        print("torchmetrics is not installed; SSDLite mAP will be reported as unavailable.")
        return None

    model = _load_ssdlite(device=device, weights_path=weights_path)
    model.eval()

    from torchvision.transforms import functional as F
    def transform(img, target): return F.to_tensor(img), target

    dataset = VehicleVOCDataset('./data', image_set='test', download=False, transforms=transform)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            if num_samples is not None and idx >= num_samples:
                break
            images = [img.to(device) for img in images]
            preds = model(images)
            preds = [{k: v.cpu() for k, v in pred.items()} for pred in preds]
            gt = prepare_ssdlite_targets(targets, device='cpu')
            metric.update(preds, gt)

    metrics = metric.compute()
    return float(metrics['map'].item())

def evaluate_yolov8n_map(data_yaml='./yolo_dataset/data.yaml', device='cuda',
                         model_path='yolo_runs/yolov8n_vehicles/weights/best.pt'):
    """Evaluate YOLOv8n mAP using Ultralytics validation."""
    if not os.path.exists(data_yaml):
        print(f"{data_yaml} not found; YOLO mAP will be reported as unavailable.")
        return None
    model, resolved_model_path = _load_yolo(model_path)
    print(f"Evaluating YOLO weights: {resolved_model_path}")
    metrics = model.val(data=os.path.abspath(data_yaml), imgsz=320,
                        device=0 if device == 'cuda' else 'cpu', verbose=False)
    return float(metrics.box.map)

def generate_comparative_report(ssdlite_fps, yolo_fps, ssdlite_map=None, yolo_map=None):
    rows = [
        {
            "Model": "SSDLite320",
            "Fine-tuned mAP": ssdlite_map,
            "COCO ref mAP": 0.220,
            "FPS": ssdlite_fps,
        },
        {
            "Model": "YOLOv8n",
            "Fine-tuned mAP": yolo_map,
            "COCO ref mAP": 0.373,
            "FPS": yolo_fps,
        },
    ]

    print("\n" + "="*60)
    print("Task 10: Comparative Report (Inference Speed vs Accuracy)")
    print("="*60)
    print(f"{'Model':<15} | {'Fine-tuned mAP':<15} | {'COCO ref mAP':<15} | {'FPS':<10}")
    print("-" * 75)
    for row in rows:
        fine_tuned = "N/A" if row["Fine-tuned mAP"] is None else f"{row['Fine-tuned mAP']:.3f}"
        print(f"{row['Model']:<15} | {fine_tuned:<15} | {row['COCO ref mAP']:<15.3f} | {row['FPS']:<10.2f}")
    print("="*60)
    return rows

def run_speed_accuracy_experiment(device='cuda', num_samples=100, map_samples=None):
    ssdlite_fps = benchmark_ssdlite(device=device, num_samples=num_samples)
    yolo_fps = benchmark_yolov8n(device=device, num_samples=num_samples)
    ssdlite_map = evaluate_ssdlite_map(device=device, num_samples=map_samples)
    yolo_map = evaluate_yolov8n_map(device=device)
    return generate_comparative_report(ssdlite_fps, yolo_fps, ssdlite_map, yolo_map)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Benchmarking speeds. This might take a few moments...\n")
    run_speed_accuracy_experiment(device=device, num_samples=100)
