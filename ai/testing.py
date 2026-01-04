import multiprocessing
import time
from ultralytics import YOLO
import traceback

# --- 1. Define the Worker Function ---
# This runs completely isolated in its own memory space
def detect_worker(model_path, model_type, shared_fps_counter, duration):
  import cv2 
  import os
  
  try:
    frames = []

    for image_name in os.listdir(os.path.join('a9_dataset/yolo_data/test', 'images')):
      frames.append(cv2.imread(os.path.join('a9_dataset/yolo_data/test', 'images', image_name)))
    
    # A. Load Model INSIDE the process
    if model_type == 'yolo':
      model = YOLO(model_path, task='detect')
      def run_inference(frame):
        model(frame, verbose=False)
            
    elif model_type == 'classical':
      from models.classical_model import ClassicDetector
      model = ClassicDetector('training/svm_sstw/scaler.onnx', 'training/svm_sstw/ovr_linear_svc.onnx')

      def run_inference(frame):
        model(frame)

    # B. Run Inference Loop
    start_time = time.time()
    while True:
      if time.time() - start_time > duration:
        break
      
      for frame in frames:
        run_inference(frame)
      
        with shared_fps_counter.get_lock():
          shared_fps_counter.value += 1
  except Exception:
    traceback.print_exc()

# --- 2. The Runner Function ---
def run_benchmark(label, model_path, model_type, timeout=10):
    shared_fps = multiprocessing.Value('i', 0)
    
    p = multiprocessing.Process(target=detect_worker, 
                                args=(model_path, model_type, shared_fps, timeout))
    p.start()
    
    print(f"[{label}] Benchmarking for {timeout} seconds...")
    p.join(timeout)

    if p.is_alive():
      p.kill()
      p.join()
    
    final_fps = shared_fps.value / timeout
    print(f"✅ {label} Finished. Average FPS: {final_fps:.2f}")
    return final_fps

# --- 3. Main Execution ---
if __name__ == '__main__':
    yolo_fp16_path = 'training/yolo_sstw/weights/best-fp16.onnx'
    yolo_fp32_path = 'training/yolo_sstw/weights/best-fp32.onnx'

    # Run
    fps16 = run_benchmark("YOLO FP16", yolo_fp16_path, 'yolo', timeout=5)
    fps32 = run_benchmark("YOLO FP32", yolo_fp32_path, 'yolo', timeout=5)
    fps_cl = run_benchmark("Classical", '', 'classical', timeout=60)
    
    # Comparison
    print(f"\n--- Final Results ---")
    print(f"YOLO FP16: {fps16:.2f}")
    print(f"YOLO FP32: {fps32:.2f}")
    print(f"Classical: {fps_cl:.2f}")