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
print(checkpoint['epoch'], checkpoint['jaccard'])
net.eval()
clip_length = 4
image_val_data = sorted(glob.glob('../val/images/*'))
mask_val_data = sorted(glob.glob('../val/masks_nao/*'))
flow_val_data = sorted(glob.glob('../val/flow/*'))

test_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=16, num_workers=1)

count = 0
jaccards = []
_, _, uid_to_obj = cleanup_obj(image_val_data, ['1'], filename='EPIC_train_action_labels.csv')
obj_to_jaccard = {}

with torch.no_grad():
    for batch_idx, (test_images, test_labels) in enumerate(test_loader):
        #flow = np.load(flow_val_data[count])[0].reshape(128, 228, 2)
        output = net(Variable(test_images).cuda().to(device))

        n = len(test_images.data.cpu().numpy())
        for b_idx in range(n):
            out_probs_cont = output[b_idx].data.cpu().numpy().reshape(-1)
            output_mask = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
            gt_mask = test_labels[b_idx].reshape(-1,).data.cpu().numpy()

            jaccard = jsc(gt_mask, output_mask)
            jaccards.append(jaccard)

            uid = image_val_data[count].split('_')[0].split('/')[3]
            #print(jaccard, uid_to_obj[uid])
            obj = int(uid_to_obj[uid])
            if obj not in obj_to_jaccard.keys():
                obj_to_jaccard[obj] = [jaccard]
            else:
                obj_to_jaccard[obj].append(jaccard)
            count += 1
#auc = roc_auc_score(np.ones(len(jaccards)), jaccards)

print(np.mean(np.array(jaccards)))
for obj in sorted(obj_to_jaccard.keys()):
    print(obj, len(obj_to_jaccard[obj]), np.mean(np.array(obj_to_jaccard[obj])))
