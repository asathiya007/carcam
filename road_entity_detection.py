import numpy as np
import cv2
import torch 
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from math_utils import MathUtils
from object_detection_models import *
from object_detection_utils import *

# configure object detection model 
config_path = './config/yolov3.cfg'
weights_path = './config/yolov3.weights'
class_path = './config/coco.names'
image_size = 416
conf_thres = 0.8
nms_thres = 0.4

# load model and weights
model = Darknet(config_path, img_size=image_size)
model.load_weights(weights_path)
if torch.cuda.is_available():
    model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.FloatTensor

# road entity detection class 
class RoadEntityDetection:
    # road entities to detect
    road_entities = set([
        'person', 
        'bicycle', 
        'car', 
        'truck', 
        'motorcycle', 
        'bus', 
        'train', 
        'boat', 
        'skis',
        'snowboard'
        'skateboard',
        'surfboard'
    ])

    # preprocess image 
    def _preprocess_image(image):
        ratio = min(image_size / image.size[0], image_size / image.size[1])
        image_width = round(image.size[0] * ratio)
        image_height = round(image.size[1] * ratio)
        image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Pad(
                (
                    max(int((image_height - image_width) / 2), 0), 
                    max(int((image_width - image_height) / 2), 0), 
                    max(int((image_height - image_width) / 2), 0),
                    max(int((image_width - image_height) / 2), 0)
                ), 
                (128, 128, 128)
            ),
            transforms.ToTensor()
         ])
        image_tensor = image_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_image = Variable(image_tensor.type(Tensor))
        return input_image
    
    # find road entities in image 
    def _find_road_entities(image):
        input_image = RoadEntityDetection._preprocess_image(image)
        with torch.no_grad(): 
            detections = model(input_image)
            detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
        return detections[0]
    
    # detect road entities in frame
    def detect_road_entities(frames, lane_lines):
        road_entity_detected_frames = []
        frame_number = 0
        for frame in frames: 
            image = Image.fromarray(frame)
            detections = RoadEntityDetection._find_road_entities(image)
            image = np.array(image)
            padding_x = max(image.shape[0] - image.shape[1], 0) * (image_size / max(image.shape))
            padding_y = max(image.shape[1] - image.shape[0], 0) * (image_size / max(image.shape))
            unpadded_height, unpadded_width = image_size - padding_y, image_size - padding_x
            if detections is not None:
                tracked_road_entities = detections.cpu().detach().numpy()
                unique_labels = detections[:, -1].cpu().unique()
                num_class_preds = len(unique_labels.detach().cpu().numpy())
                for x1, y1, x2, y2, _, _, class_pred in tracked_road_entities: 
                    box_height = int(((y2 - y1) / unpadded_height) * image.shape[0])
                    box_width = int(((x2 - x1) / unpadded_width) * image.shape[1])
                    y1 = int(((y1 - padding_y // 2) / unpadded_height) * image.shape[0])
                    x1 = int(((x1 - padding_x // 2) / unpadded_width) * image.shape[1])
                    class_name = classes[int(class_pred)]
                    if class_name in RoadEntityDetection.road_entities: 
                        box = (x1, y1, box_width, box_height)
                        collision_risk = MathUtils.line_box_intersect(lane_lines[frame_number], box)
                        if collision_risk:
                            color = (0, 0, 255, 1.0)
                        else: 
                            color = (103, 230, 114, 1.0)
                        cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + box_height), color, 4)
                        cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(class_name) * 19, y1), color, -1)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            road_entity_detected_frames.append(frame)
            frame_number += 1
        return road_entity_detected_frames            