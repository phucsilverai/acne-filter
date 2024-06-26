import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection
class InsighttFace():
    def __init__(self):
        self.model = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
        # self.model = FaceAnalysis(model_pack_name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, img_path):
        img = ins_get_image(img_path.split(".")[0])
        faces = self.model.get(img)
        # Define bounding box coordinates (xmin, ymin, width, height)
        bounding_boxes = [faces[i]['bbox'] for i in range(len(faces))] # Example bounding boxes
        scores = [faces[i]['det_score'] for i in range(len(faces))] # Example scores
        return bounding_boxes, scores

    def visualize(self, bounding_boxes, image_path):
        # Load image
        image = Image.open(image_path)

        # Plot image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)

        # Create a Rectangle patch for each bounding box
        for bbox in bounding_boxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]), linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.axis('off')
        plt.show()

class FaceXLib():
    def __init__(self, half=False):
        # half is fp16
        model_name = 'retinaface_resnet50'      # or 'retinaface_mobile0.25'
        self.model = init_detection_model(model_name, half=half)

    def __call__(self, img_path):
        img = cv2.imread(img_path)  # BGR
        with torch.no_grad():
            bboxes = self.model.detect_faces(img, conf_threshold=0.5)
            # x0, y0, x1, y1, confidence_score, five points (x, y)
            bounding_boxes = [bbox[:4] for bbox in bboxes]
            scores = [bbox[4] for bbox in bboxes]
        return bounding_boxes, scores

    def visualize(self, bounding_boxes, image_path):
        # Load image
        image = Image.open(image_path)

        # Plot image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)

        # Create a Rectangle patch for each bounding box
        for bbox in bounding_boxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]), linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.axis('off')
        plt.show()