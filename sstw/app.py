from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your PyTorch model
MODEL_PATH = 'model.pt'  # Change this to your model filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print(f"Warning: Model file '{MODEL_PATH}' not found. Please place your .pt file in the current directory.")

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process', methods=['POST'])
def process_image():
    if not model_loaded:
        return {'error': 'Model not loaded'}, 500
    
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    try:
        # Load image
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess image (adjust based on your model's requirements)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process output (this depends on your model's output format)
        # For image-to-image models, convert tensor back to PIL Image
        if output.dim() == 4:  # Batch dimension present
            result_tensor = output[0].cpu()
        else:
            result_tensor = output.cpu()
        
        # Denormalize and convert to image
        result_tensor = result_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        result_tensor = torch.clamp(result_tensor, 0, 1)
        
        result_image = transforms.ToPILImage()(result_tensor)
        
        # Save result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        result_image.save(result_path)
        
        return {'success': True, 'message': 'Image processed successfully'}
    
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/result')
def get_result():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    return {'error': 'No result available'}, 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)