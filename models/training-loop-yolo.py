import os 
import argparse
import logging 
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, formaat='%(asctime)s - %(levelname)s - %(message)s')

def train_yolo_loop(
    data_yaml_path,
    model_name='yolov8n.pt',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='0', # GPU
    project_name='computer_vision_aol',
    exp_name='exp1'
):
    """
    Runs a YOLO training loop for research purposes.
    
    Args:
        data_yaml_path (str): Path to your local data.yaml file.
        model_name (str): YOLO model variant (n, s, m, l, x) or path to weights.
        epochs (int): Number of training epochs.
        imgsz (int): Input image size.
        batch_size (int): Batch size.
        device (str): GPU device ID (e.g., '0', '0,1') or 'cpu'.
        project_name (str): Name of the project directory.
        exp_name (str): Name of the specific experiment run.
    """
    # 1. Validation Check
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Could not find data config at: {data_yaml_path}")

    logging.info(f"Starting training for experiment: {exp_name}")
    logging.info(f"Model: {model_name} | Epochs: {epochs} | Device: {device}")

    # 2. Load the Model
    # Ppre-trained model (recommended) or a YAML to build from scratch
    try:
        model = YOLO(model_name) 
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        return

    # 3. Train
    # The .train() method handles the entire loop, logging, and checkpointing
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name=exp_name,
        exist_ok=True,       # Overwrite existing experiment folder if it exists
        pretrained=True,     # Use transfer learning
        optimizer='auto',    # Ultralytics will choose SGD or AdamW automatically
        verbose=True,        # Print detailed logs
        plots=True,          # Save training plots (confusion matrix, loss curves)
        save=True            # Save checkpoints
    )

    logging.info(f"Training completed. Results saved to {project_name}/{exp_name}")

    # 4. Validate (Optional but recommended)
    logging.info("Running final validation...")
    metrics = model.val()
    logging.info(f"Final mAP50-95: {metrics.box.map}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model Training Loop")
    
    # Essential Arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='yolov8n.pt, yolov8s.pt, etc.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    
    # Experiment Tracking
    parser.add_argument('--project', type=str, default='runs/train', help='Parent folder for results')
    parser.add_argument('--name', type=str, default='custom_run', help='Specific experiment name')
    
    args = parser.parse_args()

    train_yolo_loop(
        data_yaml_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        project_name=args.project,
        exp_name=args.name
    )

"""
CARA PAKE:
GANTI dataset/data.yaml dulu sblm run
python train.py --data path/to/your/data.yaml --model yolov8s.pt --epochs 100 --name experiment_01
"""