import torch
from time_maps import FCN8s, CustomDataset

#from time_maps_flow import CustomDataset
import torch.optim as optim
import glob
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from evaluations.utils import *

def write_masks(net, test_loader, image_folder):
    i = 0
    image_data = sorted(glob.glob('./' + image_folder + '/images/*'))
    dir = './' + image_folder + '/time_map_data_rgb/'
    losses = []
    with torch.no_grad():
        for batch_idx, (test_images, target) in enumerate(test_loader):
            a = Variable(test_images).cuda()
            output = net(a)
            loss_1 = loss_seg_fn(output[:,0].reshape(-1,).to(device), target[:,0].reshape(-1,).to(device))
            loss_2 = l1(output[:,1].reshape(-1,).to(device), target[:,1].reshape(-1,).to(device))
            #print(loss_1, loss_2)
            if not np.isnan(loss_2.item()):
                losses.append(loss_1.item() + 0.2 * loss_2.item())
            #cv2.imwrite(dir + '/' + filename + '.png', 127*seg_mask_nao)
            #if i == 50:
            #    break
            for b_idx in range(16):
                filename = image_data[i].replace('./' + image_folder + '/images/', '').replace('.png', '')

                seg_mask = output.data.cpu().numpy()[b_idx][0]
                seg_mask_nao = output.data.cpu().numpy()[b_idx][1]

                gt_mask = np.zeros((128, 228))

                sampled_cont = (np.random.rand(128, 228) < seg_mask).astype(int)
                res = np.where(sampled_cont == 1)

                seg_mask_nao[res] = 0.0
                np.save(dir + '/' + filename, seg_mask_nao)
                i += 1
        print(np.mean(np.array(losses)))

num_classes = 2
device = torch.device("cuda")
net = FCN8s(num_classes).to(device)
checkpoint = torch.load('./weights/time_maps_rgb.pt')
print(checkpoint['loss'])
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

#image_data = sorted(glob.glob('./train/images/*'))
#mask_cont_data = sorted(glob.glob('./train/masks_cont/*'))
#mask_nao_data = sorted(glob.glob('./train/masks_nao/*'))

image_val_data = sorted(glob.glob('./val/images/*'))
mask_nao_data = sorted(glob.glob('./val/masks_nao/*'))
mask_cont_data = sorted(glob.glob('./val/masks_cont/*'))

#train_save_dataset = CustomDataset(image_data, mask_cont_data, mask_nao_data, train=True)
#train_save_loader = torch.utils.data.DataLoader(train_save_dataset, batch_size=1, shuffle=False, num_workers=1)

val_save_dataset = CustomDataset(image_val_data, mask_nao_data, mask_cont_data, train=True)
val_save_loader = torch.utils.data.DataLoader(val_save_dataset, batch_size=16, shuffle=False, num_workers=1)

#write_masks(net, train_save_loader, 'train')
write_masks(net, val_save_loader, 'val')
