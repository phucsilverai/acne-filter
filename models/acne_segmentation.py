"""
Download the model at
https://storage.googleapis.com/snapedit-vuhoang/vu_models/face_acne_0301/face_acne_segmenter_ISDNet_v1.pth
"""

from PIL import Image
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import time

def overlay_mask(img: np.array, mask: np.array, opacity: float = 0.5, mask_color: tuple = (255, 0, 0)) -> np.array:
    overlay = np.zeros_like(img)
    overlay[mask == 1] = mask_color
    output = cv2.addWeighted(img, 1 - opacity, overlay, opacity, 0)
    return output


class FaceAcneSegmenter():
    def __init__(self, weight_path: str, device: str):
        self.device = device
        self.model = torch.jit.load(weight_path).to(self.device)
        self.mean = np.array([123.675, 116.28, 103.53])[None, None, :]
        self.std = np.array([58.395, 57.12, 57.375])[None, None, :]

    def __call__(self, img: np.array) -> np.array:
        h, w = img.shape[:2]
        if w > h:
            new_w = 1024
            new_h = int(1024 / w * h)
        else:
            new_h = 1024
            new_w = int(1024 / h * w)
        new_h = ((new_h - 1) // 4 + 1) * 4
        new_w = ((new_w - 1) // 4 + 1) * 4
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img = (img.astype(np.float32) - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(img).squeeze(0).cpu()
            mask = torch.argmax(out, dim=0).numpy().astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask