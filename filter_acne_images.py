"""
This code use facexlib to crop the face from the image and 
then use LLaVA model to classify whether the face has acne or not.
"""

import importlib
import sys
import os
import time
from tqdm import tqdm

import torch
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np

sys.path.append('./models')

import acne_classification
import face_detection

# Load face detection model
face_detector = face_detection.FaceXLib()

# Load acne classification model
llava = acne_classification.LLaVA()

# Load data
img_root = './data/positive'

img_files = sorted(os.listdir(img_root))
img_files = [file for file in img_files if (
    file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(".png")
)]

# img_files = img_files[:100]

# Delete and create acne and non-acne folders
os.system('rm -rf ./data/acne')
os.system('rm -rf ./data/non-acne')
os.system('mkdir ./data/acne')
os.system('mkdir ./data/non-acne')
print('Created acne and non-acne folders')

acne_folder = './data/acne'
non_acne_folder = './data/non-acne'

question = "does this face have acne? only answer yes or no. no explanation or talk"

crop_face = True

for file in tqdm(img_files):
    img_path = os.path.join(img_root, file)

    if "dermnet" in img_path:
        # Copy the file to acne folder
        os.system(f'cp {img_path} {acne_folder}')
        continue

    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    img = np.array(image)

    if crop_face:
        faces_bboxes, scores = face_detector(img_path)
        if len(faces_bboxes) == 0:
            continue

        else:
            # Crop the face from image
            x1, y1, x2, y2 = faces_bboxes[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1) # to prevent negative index
            if scores[0] > 0.98:
                img = img[y1:y2, x1:x2]
            else:
                continue
            
    _, output_cropped = llava.caption_image(Image.fromarray(img), question)

    if output_cropped.lower() == 'yes':
        # Copy the file to acne folder
        os.system(f'cp {img_path} {acne_folder}')
    else:
        # Copy the file to non-acne folder
        os.system(f'cp {img_path} {non_acne_folder}')