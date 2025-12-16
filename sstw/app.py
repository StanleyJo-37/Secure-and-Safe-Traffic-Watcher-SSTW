from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max (adjust if needed)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model settings
MODEL_PATH = 'model.pt'  # put your .pt file here or change this path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Attempt to load model
model_loaded = False
model = None
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model_loaded = True
    print(f"Loaded model from {MODEL_PATH} on {device}")
except FileNotFoundError:
    model_loaded = False
    print(f"Warning: Model file '{MODEL_PATH}' not found. Place it in the project folder.")
except Exception as e:
    model_loaded = False
    print(f"Warning: Could not load model: {e}")

# Default transform (customize to fit your model)
MODEL_INPUT_SIZE = (256, 256)  # (H, W) - adjust to your model
preprocess = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def postprocess_tensor(tensor):
    # Expect tensor shape [C, H, W] in range normalized as above.
    # Denormalize and convert to uint8 numpy (H, W, BGR) for OpenCV writer.
    if isinstance(tensor, torch.Tensor):
        t = tensor.clone().detach().cpu()
    else:
        raise ValueError("postprocess expects a torch.Tensor")

    # If single channel, repeat to make 3 channels
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)

    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = t * std + mean
    t = t.clamp(0,1)
    # Convert to HWC numpy uint8 RGB
    img = (t * 255).to(torch.uint8).permute(1,2,0).numpy()
    # Convert RGB -> BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process_image', methods=['POST'])
def process_image():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # Expect output shape [B, C, H, W] or [B, ...]
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]
        if output.dim() == 4:
            result_tensor = output[0]
        else:
            result_tensor = output

        result_img = postprocess_tensor(result_tensor)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        cv2.imwrite(result_path, result_img)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result_image')
def get_result_image():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    return jsonify({'error': 'No result available'}), 404

@app.route('/process_video', methods=['POST'])
def process_video():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded video
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return jsonify({'error': 'Unable to open uploaded video'}), 500

    # Get input properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output settings - we'll write mp4
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Process frames
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Convert BGR->RGB PIL for transforms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Preprocess and inference
            input_tensor = preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)

            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            if output.dim() == 4:
                result_tensor = output[0]
            else:
                result_tensor = output

            # Convert to numpy BGR and resize back to original size
            result_img = postprocess_tensor(result_tensor)
            # result_img is BGR in model size; resize to original video frame size
            result_resized = cv2.resize(result_img, (width, height), interpolation=cv2.INTER_LINEAR)
            out.write(result_resized)

        cap.release()
        out.release()
        return jsonify({'success': True})
    except Exception as e:
        cap.release()
        out.release()
        return jsonify({'error': str(e)}), 500

@app.route('/result_video')
def get_result_video():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_video.mp4')
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='video/mp4', as_attachment=False)
    return jsonify({'error': 'No result available'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)