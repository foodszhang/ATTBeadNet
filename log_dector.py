from math import sqrt
import numpy as np
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray, gray2rgb
import skimage as ski
import cv2   # pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
import pandas as pd


import matplotlib.pyplot as plt
#from luminance_equalization import luminance_equalization
init_min_sigma = 1.5
init_max_sigma = 2.8
#init_min_sigma = 1
#init_max_sigma = 1.11

num_sigma = 30

def blob_detector(filename, result_dir):
    image_gray = ski.io.imread(filename)
    #image_gray = ski.io.imread('./bg/10.tif')
    # normlize image_gray to 0-255
    #image_gray = luminance_equalization(image_gray, 10)

    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    min_sigma = init_min_sigma
    max_sigma = init_max_sigma
    blobs_log = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma,  threshold=0.1, overlap=0)
    sigma = blobs_log[:, 2]
    unique, counts = np.unique(sigma, return_counts=True)
    while max_sigma - min_sigma > 0.2:
        blobs_log = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=0.05, overlap=0.4)
        #blobs_log = blob_doh(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=0.05, overlap=0.4)
        sigma = blobs_log[:, 2]
        unique, counts = np.unique(sigma, return_counts=True)
        times = list(zip(counts, unique))
        times.sort(reverse=True)
        mid_sigma = times[0][1]
        diff = max_sigma - min_sigma
        max_sigma = mid_sigma + diff / num_sigma
        min_sigma = mid_sigma - diff / num_sigma

    blobs_log = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=0.05, overlap=0.4)
    sigma = blobs_log[:, 2]
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    pure_img = image_gray.copy()
    pure_img = ski.exposure.rescale_intensity(pure_img, out_range=(0, 255))
    pure_img = ski.color.gray2rgb(pure_img)

    os.makedirs(result_dir, exist_ok=True)
    index = filename.split('/')[-1].split('.')[0]
    for blob in blobs_log:
        y, x, r = blob
        rr,cc = ski.draw.circle_perimeter(int(y),int(x), int(r), shape=pure_img.shape))
        pure_img[rr,cc] = [222,84,196]
    ski.io.imsave(f'{result_dir}{index}.png', pure_img.astype(np.uint8))

    return blobs_log
    

