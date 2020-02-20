import sys
import torch
sys.path.append("..")
from next_active_object import FCN8s, CustomDataset
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

image_val_data = sorted(glob.glob('../val/images/*'))
mask_val_data = sorted(glob.glob('../val/masks_nao/*'))

test_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16, num_workers=1)

center = np.zeros((128, 228))
r = 25

for i in range(len(center)):
    for j in range(len(center[i])):
        if (i - 64)**2 + (j - 114)**2 <= r**2:
            center[i][j] = 1

cv2.imwrite('./center.png', 255*center)
center = center.reshape(-1,)

count = 0
jaccards = []
aucs = []
kls = []
sims = []
tracking = [0]
contour_matching = []

for batch_idx, (test_images, test_labels) in enumerate(test_loader):
    gt_mask = test_labels.reshape(-1,).data.cpu().numpy()
    for i in range(16):
        gt_mask = test_labels[i].reshape(-1,).data.cpu().numpy()

        auc = get_judd_auc(center, gt_mask)
        aucs.append(auc)

        jaccard = jsc(gt_mask, center)
        jaccards.append(jaccard)

        kl = KLD(center, gt_mask)
        kls.append(kl)

        sim = SIM(center, gt_mask)
        sims.append(sim)

    count += 1
    if count == 10:
        break

print(np.mean(np.array(sims)))
print(np.mean(np.array(aucs)))
print(np.mean(np.array(kls)))
print(np.mean(np.array(jaccards)))
