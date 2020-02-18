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

num_classes = 2
device = torch.device("cuda")

net = FCN8s(num_classes).to(device)
checkpoint = torch.load('../weights/weights/time_maps_rgb.pt')
net.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['epoch'], checkpoint['loss'])
net.eval()
clip_length = 4
#image_val_data = sorted(glob.glob('/home/lab/all_weights/uid/train/images/*'))
#mask_val_data = sorted(glob.glob('/home/lab/all_weights/uid/train/masks/*'))
#flow_val_data = sorted(glob.glob('/home/lab/all_weights/uid/train/flow/*'))

image_val_data = sorted(glob.glob('../val/images/*'))
flow_val_data = sorted(glob.glob('../val/flow/*'))
#time_val_data = sorted(glob.glob('../val/time_map_data_rgb/*'))
mask_val_data = sorted(glob.glob('../val/masks/*'))

test_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=1)

count = 0
losses = []

with torch.no_grad():
    for batch_idx, (test_images, target) in enumerate(test_loader):
        #flow = np.load(flow_val_data[count])[0].reshape(128, 228, 2)
        output = net(Variable(test_images).cuda().to(device))
        loss_1 = loss_seg_fn(output[:,0].reshape(-1,).to(device), target[:,0].reshape(-1,).to(device))
        loss_2 = l1(output[:,1].reshape(-1,).to(device), target[:,1].reshape(-1,).to(device))
        #print(loss_1, loss_2)
        if not np.isnan(loss_2.item()):
            losses.append(loss_1.item() + 0.2 * loss_2.item())
    #    if image_val_data[count].split('_')[2] == image_val_data[count-1].split('_')[2]:
    #        coh_err = coherence_error(flow, prev_prediction, output_mask)
    #        tracking.append(coh_err)

    #    contour_pred, contour_gt = find_contours(output_mask, 1), find_contours(gt_mask, 1)
        #print(jaccard, kl, sim)
        #prev_prediction = output_mask
        count += 1
        if count == 100:
            break

#auc = roc_auc_score(np.ones(len(jaccards)), jaccards)

print(np.mean(np.array(losses)))
