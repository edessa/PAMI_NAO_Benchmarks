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
center = np.tile(center, (16,))


count = 0
jaccards = []
kls = []
sims = []
tracking = [0]
contour_matching = []

for batch_idx, (test_images, test_labels) in enumerate(test_loader):
    gt_mask = test_labels.reshape(-1,).data.cpu().numpy()
    jaccard = jsc(gt_mask, center)
    jaccards.append(jaccard)
    kl = KL(normalize(center), normalize(gt_mask.reshape(-1,)))
    kls.append(kl)
    sim = histogram_intersection(gt_mask, center)
    sims.append(sim)

#    if image_val_data[count].split('_')[2] == image_val_data[count-1].split('_')[2]:
#        coh_err = coherence_error(flow, prev_prediction, output_mask)
#        tracking.append(coh_err)

#    contour_pred, contour_gt = find_contours(output_mask, 1), find_contours(gt_mask, 1)
    #print(jaccard, kl, sim)
    count += 1
    if count == 100:
        break

#auc = roc_auc_score(np.ones(len(jaccards)), jaccards)
print(np.mean(np.array(jaccards)))
print(np.mean(np.array(kls)))
print(np.mean(np.array(sims)))
