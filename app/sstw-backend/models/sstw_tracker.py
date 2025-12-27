
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2
from models.classical_model import ClassicDetector
from ultralytics import YOLO
import os
import random
import traceback

class SSTWTracker():
	def __init__(
		self,
		in_dir: str,
		out_dir: str,
		detector: ClassicDetector | YOLO
	):
		self.detector = detector
		self.tracker = DeepSort(
			max_age=20,
			n_init=2,
			embedder='clip_ViT-B/16',
			half=True,
			embedder_gpu=True,
		)
		
		self.target_classes = np.arange(7)
		self.color_palette = {}
		
		self.in_dir = in_dir
		self.out_dir = out_dir
		
		self.fourcc = cv2.VideoWriter.fourcc(*'avc1')
		
	def __call__(self, video_paths) -> list[str]:
		output_paths = []
		
		for video_path in video_paths:
			input_video_path = os.path.join(self.in_dir, video_path)
			
			filename = os.path.basename(video_path)
			output_video_path = os.path.join(self.out_dir, f'out_{filename}')
			
			cap = cv2.VideoCapture(input_video_path)
			if not cap.isOpened():
				print(f"Error opening video: {input_video_path}")
				continue
			
			frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
			
			out = cv2.VideoWriter(output_video_path, self.fourcc, frame_fps, (frame_width, frame_height))

			frame_count = 0
			try:
				while cap.isOpened():
					ret, frame = cap.read()
					
					if not ret: break
					
					results = self.detector(frame, conf=0.5, iou=0.4, imgsz=640, verbose=False)[0]
					
					detections = []
					for box in results.boxes:
						x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
						
						conf = float(box.conf[0])
						cls_id = int(box.cls[0])
						
						if cls_id in self.target_classes:
							detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
					
					tracks = self.tracker.update_tracks(detections, frame=frame)
					
					for track in tracks:
						if not track.is_confirmed(): continue
						
						track_id = track.track_id
						ltrb = track.to_ltrb()
						
						x1, y1, x2, y2 = map(int, ltrb)
						
						if track_id not in self.color_palette:
							self.color_palette[track_id] = (
								random.randint(50, 200),
								random.randint(50, 200),
								random.randint(50, 200),
							)
						
						color = self.color_palette[track_id]
						
						cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
						
						text = f"ID: {track_id}"
						text_scale = 1.5
						text_thickness = 4
						text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
																				text_scale, text_thickness)[0]

						bg_x1 = x1
						bg_y1 = max(0, y1 - text_size[1] - 10)
						bg_x2 = x1 + text_size[0] + 5
						bg_y2 = y1 - 10
						
						if bg_y1 >= 0 and bg_y2 < frame_height and bg_x2 < frame_width:
							cv2.rectangle(frame, 
														(bg_x1, bg_y1),
														(bg_x2, bg_y2),
														(255, 255, 255), -1)  # White background
					
							# Display ID with same color as bounding box
							cv2.putText(frame, text, (x1, y1 - 15), 
													cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 
													text_thickness)
					
					out.write(frame)
					
					frame_count += 1
					
					if frame_count % 30 == 0:
						print(f"Processed {frame_count} frames.")
			except Exception as e:
				print(f'Error/Interrupted Processing Video {video_path}.')
				print(f'Reason: {e}')
				traceback.print_exc()
				cap.release()
				out.release()
				continue
			finally:
				cap.release()
				out.release()
				print(f'Video saved to: {output_video_path}')
				print(f'Total frames processed: {frame_count}')
    
			output_paths.append(output_video_path)
		
		return output_paths
