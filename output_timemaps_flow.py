import torch
from time_maps_flow import FCN8s
from time_maps_flow import CustomDataset
#from time_maps_flow import CustomDataset
import torch.optim as optim
import glob
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

def write_masks(net, test_loader, image_folder):
    i = 0
    image_data = sorted(glob.glob('./' + image_folder + '/images/*'))
    dir = './time_map_data_flow/'
    for batch_idx, (test_images, test_labels) in enumerate(test_loader):
        filename = image_data[i].replace('./' + image_folder + '/images/', '').replace('.png', '')
        a = Variable(test_images).cuda()
        out_labels = net(a).data.cpu().numpy()
        seg_mask = out_labels[0][0]
        seg_mask_nao = out_labels[0][1]

        gt_mask = np.zeros((128, 228))

        sampled_cont = (np.random.rand(128, 228) < seg_mask).astype(int)
        res = np.where(sampled_cont == 1)

        seg_mask_nao[res] = 0.0
        np.save(dir + image_folder + '/' + filename, seg_mask_nao)
        i += 1

num_classes = 2
clip_length = 3
device = torch.device("cuda")
net = FCN8s(num_classes).to(device)
#checkpoint = torch.load('./best_weights/time_maps_flow/time_maps_rgb_4.0.pt')

net.load_state_dict(checkpoint)
net.eval()

image_data = sorted(glob.glob('./train/images/*'))
mask_data = sorted(glob.glob('./train/masks/*'))

image_val_data = sorted(glob.glob('./val/images/*'))
mask_val_data = sorted(glob.glob('./val/masks/*'))

train_save_dataset = CustomDataset(image_data, mask_data, clip_length=clip_length, train=True)
train_save_loader = torch.utils.data.DataLoader(train_save_dataset, batch_size=1, shuffle=False, num_workers=1)

val_save_dataset = CustomDataset(image_val_data, mask_val_data, clip_length=clip_length, train=True)
val_save_loader = torch.utils.data.DataLoader(val_save_dataset, batch_size=1, shuffle=False, num_workers=1)

write_masks(net, train_save_loader, 'train')
write_masks(net, val_save_loader, 'val')
