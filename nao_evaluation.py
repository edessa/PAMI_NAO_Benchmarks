import torch
from next_active_object_time_maps import FCN8s, CustomDataset #Python file to change
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

num_classes = 1
device = torch.device("cuda")

net = FCN8s(num_classes).to(device)
checkpoint = torch.load('./weights/nao_time_maps.pt')
net.load_state_dict(checkpoint)

clip_length = 4
image_val_data = sorted(glob.glob('./val/images/*'))
flow_val_data = sorted(glob.glob('./val/flow/*'))
time_val_data = sorted(glob.glob('./val/time_map_data_rgb/*'))
mask_val_data = sorted(glob.glob('./val/masks/*'))

test_loader = CustomDataset(image_val_data, time_val_data, mask_val_data, clip_length = 4, train=True)
test_loader = torch.utils.data.DataLoader(test_loader, shuffle=False, batch_size=1, num_workers=1)

count = 0
jaccards = []
kls = []
sims = []
tracking = []
contour_matching = []

for batch_idx, (test_images, test_labels) in enumerate(test_loader):
    flow = np.load(flow_val_data[count])[0].reshape(128, 228, 2)
    output = Variable(test_images).cuda()
    out_probs_cont = output[:,0].data.cpu().numpy().reshape(-1)
    output_mask = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
    gt_mask = test_labels.reshape(-1,)

    jaccard = jsc(gt_mask, output_mask)
    jaccards.append(jaccard)

    kl = entropy(pk=out_probs_cont, qk=gt_mask.reshape(-1,))
    kls.append(kl)
    sims.append(histogram_intersection(gt_mask, output_mask))

    if image_val_data[count].split('_')[2] == image_val_data[count-1].split('_')[2]:
        tracking.append(coherence_error(flow, prev_prediction, output_mask))

#    contour_pred, contour_gt = find_contours(output_mask, 1), find_contours(gt_mask, 1)

    prev_prediction = output_mask
    count += 1

auc = roc_auc_score(np.ones(len(jaccards)), jaccards)

print(auc)
