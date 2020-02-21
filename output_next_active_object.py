import torch
from next_active_object import FCN8s, CustomDataset #Python file to change
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
checkpoint = torch.load('/home/lab/Object_Split/weights/nao_overfit.pt')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

clip_length = 4
image_val_data = sorted(glob.glob('./train/images/*'))
flow_val_data = sorted(glob.glob('./train/flow/*'))
#time_val_data = sorted(glob.glob('./train/time_map_data_rgb/*'))
mask_val_data = sorted(glob.glob('./train/masks_nao/*'))

test_loader = CustomDataset(image_val_data, mask_val_data, train=False)
test_loader = torch.utils.data.DataLoader(test_loader, shuffle=False, batch_size=16, num_workers=1)

count = 0
jaccards = []
with torch.no_grad():
    for batch_idx, (test_images, test_labels) in enumerate(test_loader):
        output = net(test_images.to(device))
        out_probs_cont = output.data.cpu().numpy().reshape(-1)
        output_mask = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
        gt_mask = test_labels.reshape(-1,).data.cpu().numpy()

        jaccard = jsc(gt_mask, output_mask)
        jaccards.append(jaccard)
        print(jaccard, output.shape)
        if jaccard > 0.5:
            for b_idx in range(16):
                np.save('./train/nao_predictions/' + image_val_data[count].replace('./train/images/', '').replace('.png', ''), output[b_idx].data.cpu().numpy())
                #cv2.imwrite('./train/nao_predictions/' + image_val_data[count].replace('./train/images/', ''), cv2.imread(image_val_data))
                count += 1
