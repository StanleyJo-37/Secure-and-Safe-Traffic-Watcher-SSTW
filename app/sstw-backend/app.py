from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from models.sstw_tracker import SSTWTracker
import os
import threading
import uuid
from flask_socketio import SocketIO

BASE_URL = 'http://localhost:5000'

TEMP_VIDEOS_PATH = 'tmp-videos'
TEMP_UPLOADED_VIDEOS_PATH = os.path.join(TEMP_VIDEOS_PATH, 'uploads')
TEMP_PROCESSED_VIDEOS_PATH = os.path.join(TEMP_VIDEOS_PATH, 'processed')

ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = TEMP_UPLOADED_VIDEOS_PATH
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs(TEMP_UPLOADED_VIDEOS_PATH, exist_ok=True)
os.makedirs(TEMP_PROCESSED_VIDEOS_PATH, exist_ok=True)

print("Loading model...")
try:
    detector = YOLO('models/weights/best-fp32.onnx')
    tracker = SSTWTracker(TEMP_UPLOADED_VIDEOS_PATH, TEMP_PROCESSED_VIDEOS_PATH, detector)
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model_loaded = False
    detector = None
    tracker = None
    
@app.get('/processed/<path:filename>')
def serve_processed_video(filename):
    return send_from_directory(TEMP_PROCESSED_VIDEOS_PATH, filename)
    
def track(task_id: str, upload_filename: str):
    print(f"[{task_id}] Processing started...")
    
    tracker([upload_filename])
    
    video_url = f"{BASE_URL}/processed/out_{upload_filename}"
    socketio.emit('task_completed', {
        'task_id': task_id,
        'status': 'COMPLETED',
        'video_url': video_url,
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413

@app.post('/tracker/upload')
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No videos found.'}), 400
    
    video = request.files['video']
    task_id = str(uuid.uuid4())
    
    upload_filename = f'{task_id}_{video.filename}'
    save_path = os.path.join(TEMP_UPLOADED_VIDEOS_PATH, upload_filename)
    
    video.save(save_path)
    
    thread = threading.Thread(target=track, args=(task_id, upload_filename))
    thread.start()
    
    return jsonify({"message": "Uploaded", "task_id": task_id}), 202

if __name__ == '__main__':
    socketio.run(debug=True, port=5000)