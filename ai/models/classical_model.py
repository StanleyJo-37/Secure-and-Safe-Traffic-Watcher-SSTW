import onnxruntime as ort
import cv2
import numpy as np
import cv2
import joblib
from utils.dcp.haze_remover import HazeRemover
from utils.ltp.ltp import LTP

class Box():
	def __init__(self, bbox, label, score):
		self.xyxy = np.array([bbox])
		self.conf = [score]
		self.cls = [label]
	
class Result:
	def __init__(self, boxes, labels, scores):
		self.boxes = []

		for b, l, s in zip(boxes, labels, scores):
			self.boxes.append(Box(b, l, s))

class ClassicDetector():
	def __init__(
		self,
		scaler_path: str,
		classifier_model_path: str,
	):
		self.scaler = ort.InferenceSession(scaler_path, providers=ort.get_available_providers())
		self.clf = ort.InferenceSession(classifier_model_path, providers=ort.get_available_providers())
	
		self.scaler_input_name = self.scaler.get_inputs()[0].name
		self.clf_input_name = self.clf.get_inputs()[0].name

		self.haze_remover = HazeRemover()
		self.clahe = cv2.createCLAHE(2.0, (8, 8))
		self.patch_size = (128, 64)

		self.selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
		self.pca = joblib.load('outputs/hog_pca.joblib')
		self.hog = cv2.HOGDescriptor(self.patch_size, (16, 16), (8, 8), (8, 8), 9)
		self.ltp = LTP()

	def variance_of_laplacian(self, image):
		return cv2.Laplacian(image, cv2.CV_64F).var()

	def is_foggy(self, image):
		var_l = self.variance_of_laplacian(image)
		return var_l < 50

	def preprocess(self, img, foggy:bool=False):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray_final = cv2.medianBlur(img_gray, 3)
		
		if foggy or self.is_foggy(img_gray_final):
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_dehazed = self.haze_remover.remove_haze(img_rgb)
			img_dehazed = np.clip(img_dehazed, 0, 255).astype(np.uint8)
			
			img_gray_final = cv2.cvtColor(img_dehazed, cv2.COLOR_RGB2GRAY)
		
		img_clahe = self.clahe.apply(img_gray_final)
		
		return img_clahe.astype(np.uint8)

	def extract_features(self, img):
		# Get bounding boxes using Selective Search
		
		resized_img = cv2.resize(img, self.patch_size)
		
		if len(resized_img.shape) == 2:
				# It is already 2D (Height, Width) -> No conversion needed
				gray_image = resized_img
		elif resized_img.shape[2] == 1:
				# It is 3D but has 1 channel (Height, Width, 1) -> Squeeze it
				gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # Sometimes safe, but better to just squeeze
				gray_image = resized_img.squeeze()
		else:
				# It is BGR (Height, Width, 3) -> Convert
				gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

		feat = []

		# HOG
		hog_feat = self.hog.compute(gray_image).flatten()
		hog_feat = self.pca.transform(hog_feat.reshape(1, -1)).flatten()
		feat.extend(hog_feat)

		# LTP
		ltp_feat = self.ltp(gray_image, 10)
		feat.extend(ltp_feat.flatten())

		return np.array(feat, dtype=np.float32)

	def __call__(self, source, conf=0.5, iou=0.4, *args, **kwargs) -> list[Result]:
		results = []
		images = []
	
		if isinstance(source, np.ndarray):
			images = [source]
		elif isinstance(source, str):
			images = [cv2.imread(source)]
		elif isinstance(source, list):
			for item in source:
				if isinstance(item, str):
					images.append(cv2.imread(item))
				elif isinstance(item, np.ndarray):
					images.append(item)

		for image in images:
			if image is None: 
				results.append([])
				continue
		
			boxes = []
			labels = []
			scores = []
		
			orig_h, orig_w = image.shape[:2]
			preprocessed_image = self.preprocess(image)
			resized_img = cv2.resize(image, (640, 640))
			scale_x = orig_w / 640.0
			scale_y = orig_h / 640.0

			self.selective_search.setBaseImage(resized_img)
			self.selective_search.switchToSelectiveSearchFast()
			proposed_boxes = self.selective_search.process()

			for x, y, w, h in proposed_boxes[:1000]:
				x = int(x * scale_x)
				y = int(y * scale_y)
				w = int(w * scale_x)
				h = int(h * scale_y)
		
				patch = preprocessed_image[y:y+h, x:x+w]
		
				if patch.size == 0: continue

				features = self.extract_features(patch)
				features = features.reshape(1, -1)
				features = features.astype(np.float32)
    
				scaled_features = self.scaler.run(None, {self.scaler_input_name: features})[0]

				svm_results = self.clf.run(None, {self.clf_input_name: scaled_features})
		
				cls_id = svm_results[0][0]
				raw_score = 0.0
				if len(svm_results) > 1:
					scores_val = svm_results[1] 
					if isinstance(scores_val, np.ndarray):
						if scores_val.size > 1:
							raw_score = float(np.max(scores_val))
						else:
							raw_score = float(scores_val.item())
									
					elif isinstance(scores_val, list):
						if len(scores_val) > 1:
							raw_score = float(max(scores_val))
						elif len(scores_val) == 1:
							raw_score = float(scores_val[0])

				if cls_id != 6 and raw_score > conf:
					boxes.append([x, y, w, h])
					labels.append(cls_id)
					scores.append(float(raw_score))

			final_boxes = []
			final_labels = []
			final_scores = []
			if len(boxes) > 0:
				indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)

				if isinstance(indices, tuple):
					indices = list(indices)
				elif isinstance(indices, np.ndarray):
					indices = indices.flatten().tolist()

				for i in indices:
					idx = i if isinstance(i, int) else i[0]

					x, y, w, h = boxes[idx]

					x_min, y_min = x, y
					x_max, y_max = x + w, y + h

					final_boxes.append([x_min, y_min, x_max, y_max]) 
					final_labels.append(int(labels[idx]))
					final_scores.append(float(scores[idx]))

			result = Result(final_boxes, final_labels, final_scores)
			results.append(result)
		return results