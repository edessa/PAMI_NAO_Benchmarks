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

num_classes = 1
device = torch.device("cuda")

net = FCN8s(num_classes).to(device)
checkpoint = torch.load('../weights/nao.pt')
net.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['epoch'])
net.train()
clip_length = 4
image_val_data = sorted(glob.glob('../val/images/*'))
mask_val_data = sorted(glob.glob('../val/masks/*'))
flow_val_data = sorted(glob.glob('../val/flow/*'))

test_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16, num_workers=1)

count = 0
jaccards = []
kls = []
sims = []
tracking = [0]
contour_matching = []

with torch.no_grad():
    for batch_idx, (test_images, test_labels) in enumerate(test_loader):
        #flow = np.load(flow_val_data[count])[0].reshape(128, 228, 2)
        output = net(Variable(test_images).cuda().to(device))
        out_probs_cont = output.data.cpu().numpy().reshape(-1)
        output_mask = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
        gt_mask = test_labels.reshape(-1,).data.cpu().numpy()

        jaccard = jsc(gt_mask, output_mask)
        jaccards.append(jaccard)

        kl = KL(normalize(out_probs_cont), normalize(gt_mask.reshape(-1,)))
        kls.append(kl)
        sim = histogram_intersection(gt_mask, output_mask)
        sims.append(sim)

    #    if image_val_data[count].split('_')[2] == image_val_data[count-1].split('_')[2]:
    #        coh_err = coherence_error(flow, prev_prediction, output_mask)
    #        tracking.append(coh_err)

    #    contour_pred, contour_gt = find_contours(output_mask, 1), find_contours(gt_mask, 1)
        #print(jaccard, kl, sim)
        prev_prediction = output_mask
        count += 1
        if count == 100:
            break

auc = roc_auc_score(np.ones(len(jaccards)), jaccards)
print(auc)
print(np.mean(np.array(jaccards)))
