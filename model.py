import torch
from torch import nn
import cv2
import numpy as np
from torchvision.ops import roi_align
from typing import Tuple

class FeatureExtraction(nn.Module):
	def __init__(
		self,
  	region_proposal_search,
		pooling_size: Tuple[int, int] = (7, 7),
		hog_descriptor: cv2.HOGDescriptor = cv2.HOGDescriptor(),
		hog_win_stride: Tuple[int, int] = (8, 8),
		hog_padding: Tuple[int, int] = (8, 8),
		hog_locations: Tuple[Tuple[int, int]] = ((10, 20,),),
		**kwargs
	):
		super().__init__(**kwargs)

		self.hog_descriptor = hog_descriptor
  
		if region_proposal_search:
			self.region_proposal_search = region_proposal_search
		else:
			self.region_proposal_search = cv2.ximgproc.createEdgeBoxes()
  
		self.pooling_size = pooling_size
  
		self.hog_win_stride = hog_win_stride
		self.hog_padding = hog_padding
		self.hog_locations = hog_locations
	
	def forward(self, X):
  	# Edge and Orientation Map Extraction 
		sobel_x = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=3)
		sobel_y = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=3)

		edges, orientation_map = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

		# Bounding Boxes Extraction
		boxes = self.region_proposal_search.getBoundingBoxes(edges, orientation_map)
  
		# HOG Extraction
		hog = self.hog_descriptor.compute(
    	X,
     	self.hog_win_stride,
     	self.hog_padding,
      self.hog_locations,
    )
  
  	# Corner Extraction
		gray_img = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
		corners = cv2.goodFeaturesToTrack(gray_img, 25, 0.01, 10)
		corners = np.int32(corners)

		# RoI Features extractions
		roi_features = roi_align(X, boxes, self.pooling_size)
  
		complete_features = {
			"hog": hog.flatten(),
			"edges": edges,
			"orientation_map": orientation_map,
			"corners": corners,
			"roi": roi_features,
		}
  
		return complete_features, boxes

class DetectionHead(nn.Module):
	def __init__(
		self,
		label_num_classes: int,
		color_num_classes: int,
		feature_dim: int,
		**kwargs
	):
		super().__init__(**kwargs)
		self.label_classifier = nn.Sequential(
			nn.Linear(feature_dim, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(256, 128),
			nn.LeakyReLU(),

			nn.Linear(128, 64),
			nn.LeakyReLU(),

			nn.Linear(64, 32),
			nn.LeakyReLU(),

			nn.Linear(32, 16),
			nn.LeakyReLU(),

			nn.Linear(16, 8),
			nn.LeakyReLU(),

			nn.Linear(8, 1),
			nn.Softmax(),
		)
  
		self.color_classifier = nn.Sequential(
			nn.Linear(feature_dim, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(256, 128),
			nn.LeakyReLU(),

			nn.Linear(128, 64),
			nn.LeakyReLU(),

			nn.Linear(64, 32),
			nn.LeakyReLU(),

			nn.Linear(32, 16),
			nn.LeakyReLU(),

			nn.Linear(16, 8),
			nn.LeakyReLU(),

			nn.Linear(8, 1),
			nn.Softmax(),
		)
  
		self.bbox_regression = nn.Sequential(
			nn.Linear(feature_dim, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.1),

			nn.Linear(256, 128),
			nn.LeakyReLU(),

			nn.Linear(128, 64),
			nn.LeakyReLU(),

			nn.Linear(64, 32),
			nn.LeakyReLU(),

			nn.Linear(32, 16),
			nn.LeakyReLU(),

			nn.Linear(16, 4),
		)

	def forward(self, complete_features, bbox):
		label = torch.softmax(self.label_classifier(complete_features), dim=1)
		color = torch.softmax(self.color_classifier(complete_features), dim=1)
		bbox_deltas = self.bbox_regression(bbox)

		return label, color, bbox_deltas

class Model(nn.Module):
	def __init__(
		self,
		num_classes: int,
		feature_dim: int,
		**kwargs
	):
		super().__init__(**kwargs)

		self.feature_extraction = FeatureExtraction()
		self.detection_head = DetectionHead(num_classes, feature_dim)

	def forward(self, X):
		complete_features, bbox_deltas = self.feature_detection(X)
		label, color, bbox_deltas = self.detection_head(complete_features, bbox_deltas)
  
		bbox = []
  
		color = self.colors[color]["name"]
		label = self.categories[label]["name"]
  
		return label, color, bbox
