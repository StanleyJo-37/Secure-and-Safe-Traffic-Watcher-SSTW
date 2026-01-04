def compute_iou(box1, box2):
  x1a, y1a, x2a, y2a = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
  x1b, y1b, x2b, y2b = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
  
  intersection_width = min(x2a, x2b) - max(x1a, x1b)
  intersection_height = min(y2a, y2b) - max(y1a, y1b)
  
  if intersection_width <= 0 or intersection_height <= 0:
    return 0
  
  intersection_area = intersection_width * intersection_height

  # Calculate union area
  box1_area = box1[2] * box1[3]
  box2_area = box2[2] * box2[3]
  
  union_area = box1_area + box2_area - intersection_area

  # Calculate IoU
  iou = intersection_area / union_area
  return iou
