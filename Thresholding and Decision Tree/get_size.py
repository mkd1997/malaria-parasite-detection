import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from PIL import Image
from resizeimage import resizeimage

train_image_list_parasitized = os.listdir('./train/Parasitized/')
# print(train_image_list_parasitized)
for image in train_image_list_parasitized:
    full_path = './train/Parasitized/' + image
    print(full_path)
    save_path = './train/Parasitized/done/' + image
    if('.png' in full_path):
#     img = cv2.imread(full_path)
#     with open(full_path, 'r+b') as f:
#         with Image.open(f) as img:
#             cover = resizeimage.resize_cover(img, [100, 100])
#             cover.save(full_path, img.format)
#     resized_img = cv2.resize(img, (100, 100), interpolation=(cv2.INTER_AREA))
        fd_img = open(full_path, 'r')
        img = Image.open(fd_img)
        img = resizeimage.resize_crop(img, [120, 120])
        img.save(save_path, img.format)
        fd_img.close()