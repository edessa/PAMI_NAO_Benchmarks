import sys
import torch
sys.path.append("..")
from time_maps import FCN8s, CustomDataset
from sklearn.metrics import jaccard_score as jsc
import torch.optim as optim
from skimage.measure import find_contours
import glob
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from utils import *
import scipy.ndimage as ndimage


image_val_data = sorted(glob.glob('../val/images/*'))
mask_val_data = sorted(glob.glob('../val/masks/*'))
time_map_data = sorted(glob.glob('../val/time_map_data_rgb/*'))

for i in range(len(image_val_data)):
    image = cv2.imread(image_val_data[i])
    time_map = np.load(time_map_data[i])
    time_map = np.array((255.0/np.max(time_map))*time_map, dtype=np.uint8)
    #time_map = ndimage.gaussian_filter(time_map, sigma=(5, 5, 0), order=0)
    heatmap_time = cv2.resize(cv2.applyColorMap(time_map, cv2.COLORMAP_BONE), (456, 256), interpolation=cv2.INTER_AREA)
    print(heatmap_time)
    overlay = cv2.addWeighted(heatmap_time, 0.9, image, 0.1, 0)
    cv2.imshow("Image", image)
    cv2.imshow("Time Map", overlay)
    cv2.waitKey(0)
    cv2.imwrite('../heatmap_output/' + str(i) + '_image.png', image)
    cv2.imwrite('../heatmap_output/' + str(i) + '_times.png', overlay)
