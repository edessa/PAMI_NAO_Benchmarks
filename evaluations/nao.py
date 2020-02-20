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

net.train()
clip_length = 4
image_val_data = sorted(glob.glob('../val/images/*'))
mask_val_data = sorted(glob.glob('../val/masks_nao/*'))
flow_val_data = sorted(glob.glob('../val/flow/*'))

test_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16, num_workers=1)

count = 0
jaccards = []
aucs = []
kls = []
sims = []
tracking = [0]
contour_matching = []

with torch.no_grad():
    for batch_idx, (test_images, test_labels) in enumerate(test_loader):
        #flow = np.load(flow_val_data[count])[0].reshape(128, 228, 2)
        output = net(Variable(test_images).cuda().to(device))

        for i in range(16):
            out_probs_cont = output[i].data.cpu().numpy().reshape(-1)
            output_mask = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
            gt_mask = test_labels[i].reshape(-1,).data.cpu().numpy()

            auc = get_judd_auc(output_mask, gt_mask)
            aucs.append(auc)

            jaccard = jsc(gt_mask, output_mask)
            jaccards.append(jaccard)

            kl = KLD(out_probs_cont, gt_mask)
            kls.append(kl)

            sim = SIM(out_probs_cont, gt_mask)
            sims.append(sim)


        count += 1
        if count == 10:
            break

print(np.mean(np.array(sims)))
print(np.mean(np.array(aucs)))
print(np.mean(np.array(kls)))
print(np.mean(np.array(jaccards)))
