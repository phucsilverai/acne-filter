""" 
This code uses facexlib to detect the faces from the image. 
This serves as a pre-processing step to filter out images that do not contain faces.

It saves the images with faces to the positive folder and 
the images without faces to the negative folder.
"""
import importlib
import sys
import os
import time
import random

import torch
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm

import face_detection

face_detector = face_detection.FaceXLib()

img_root = '/home/phucle/code/data/clean_data/'

img_files = sorted(os.listdir(img_root))
img_files = [file for file in img_files if (
    file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(".png")
)]

def has_face(img_name, threshold=0.98):
    img_path = os.path.join(img_root, img_name)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img = np.array(img)

    faces_bboxes, scores = face_detector(img_path)
    
    # If the scores[0] > 0.98 then append it to results.txt
    if scores and scores[0] > threshold:
        with open('positive.txt', 'a') as f:
            f.write(img_name + '\n')
            # Copy the image to positive folder
            os.system(f'cp {img_path} positive/')
    else:
        with open('negative.txt', 'a') as f:
            f.write(img_name + '\n')
            os.system(f'cp {img_path} negative/')


with open('positive.txt', 'w') as f:
    f.write('')
print("Create new positive.txt")

with open('negative.txt', 'w') as f:
    f.write('')
print("Create new negative.txt")

# Create a positive and negative folder, if exists then remove it and create a new one
if os.path.exists('positive'):
    os.system('rm -rf positive')
os.mkdir('positive')
print("Create new positive folder")

if os.path.exists('negative'):
    os.system('rm -rf negative')
os.mkdir('negative')
print("Create new negative folder")

# Check the quality of each image
for file in tqdm(img_files):
    has_face(file, threshold=0.98)