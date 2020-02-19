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
    dir = './' + image_folder + '/time_map_data_flow/'
    with torch.no_grad():
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
            np.save(dir + filename, seg_mask_nao)
            i += 1

num_classes = 2
clip_length = 3
device = torch.device("cuda")
net = FCN8s(num_classes).to(device)
checkpoint = torch.load('./weights/time_maps_flow.pt')

net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

image_data = sorted(glob.glob('./train/images/*'))
flow_data = sorted(glob.glob('./train/flow/*'))
mask_nao_data = sorted(glob.glob('./train/masks_nao/*'))
mask_cont_data = sorted(glob.glob('./train/masks_cont/*'))

image_val_data = sorted(glob.glob('./val/images/*'))
flow_val_data = sorted(glob.glob('./val/flow/*'))
mask_nao_val_data = sorted(glob.glob('./val/masks_nao/*'))
mask_cont_val_data = sorted(glob.glob('./val/masks_cont/*'))

train_dataset = CustomDataset(image_data, flow_data, mask_nao_data, mask_cont_data, clip_length=clip_length, train=True)
val_dataset = CustomDataset(image_val_data, flow_val_data, mask_nao_val_data, mask_cont_val_data, clip_length=clip_length, train=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=1)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=1)

write_masks(net, train_loader, 'train')
write_masks(net, test_loader, 'val')
