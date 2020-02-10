import torch
from time_maps import FCN8s
from time_maps import CustomDataset
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
    dir = './' + image_folder + '/time_map_data_rgb/'
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
        cv2.imwrite(dir + '/' + filename + '.png', 127*seg_mask_nao)
        #np.save(dir + '/' + filename, seg_mask_nao)
        i += 1

num_classes = 2
device = torch.device("cuda")
net = FCN8s(num_classes).to(device)
checkpoint = torch.load('./best_weights/time_maps_rgb/time_maps_rgb_98.pt')

net.load_state_dict(checkpoint)
net.eval()

image_data = sorted(glob.glob('./train/images/*'))
mask_data = sorted(glob.glob('./train/masks/*'))

image_val_data = sorted(glob.glob('./val/images/*'))
mask_val_data = sorted(glob.glob('./val/masks/*'))

train_save_dataset = CustomDataset(image_data, mask_data, train=True)
train_save_loader = torch.utils.data.DataLoader(train_save_dataset, batch_size=1, shuffle=False, num_workers=1)

val_save_dataset = CustomDataset(image_val_data, mask_val_data, train=True)
val_save_loader = torch.utils.data.DataLoader(val_save_dataset, batch_size=1, shuffle=False, num_workers=1)

write_masks(net, train_save_loader, 'train')
write_masks(net, val_save_loader, 'val')
