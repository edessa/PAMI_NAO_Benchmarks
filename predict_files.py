import torch
from next_active_object import FCN8s, CustomDataset
import torch.optim as optim
import glob
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

num_classes = 2
device = torch.device("cuda")

net = FCN8s(num_classes).to(device)
#checkpoint = torch.load('./weights/mse_weights.pt')
#net.load_state_dict(checkpoint)
print(net)
image_data = sorted(glob.glob('/home/lab/Object_Split/train/images/*'))
mask_data = sorted(glob.glob('/home/lab/Object_Split/train/masks/*'))
nao_data = sorted(glob.glob('/home/lab/Object_Split/train/nao_predictions/*'))
test_dataset = CustomDataset(image_data, mask_data, train=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
i = 0
j = 0
count = 0
#net.train()
prev_start = 0
cm = plt.get_cmap('gist_rainbow')

for batch_idx, (test_images, test_labels) in enumerate(test_loader):
    filename = image_data[count]
    print(filename)
    start = int(filename.split('_')[2])
    end = int(filename.split('_')[3])
    if start != prev_start:
        if i > 0:
            gif_list[0].save('./mosaics_all/' + str(i) + '.gif', save_all=True, duration=40, loop=0, append_images=gif_list[1:])
        j = 0
        i += 1
        gif_list = []
        #im = Image.new('RGB', (228*int(end-start), 128*3))
    a = Variable(test_images).cuda()

    print(count)
    #seg_mask = np.array(cv2.imread(dir  + str(count) + '.png', cv2.IMREAD_GRAYSCALE)) / 255.0
    #nao_mask = np.array(cv2.imread(dir  + str(count) + '.png', cv2.IMREAD_GRAYSCALE)) / 255.0
    #print(seg_mask.shape, nao_mask.shape)
    seg_mask = np.load(nao_data[count])
    nao_mask = np.load(nao_data[count])
    #seg_mask = out_labels[0][0]
    #nao_mask = out_labels[0][1]
    overall_image = Image.new('RGB', (228, 128*3))
    img = test_images[0].data.cpu().numpy().transpose(1, 2, 0).astype('uint8')
    input_image = Image.fromarray(img, 'RGB')

    heatmap_seg = cv2.applyColorMap((255*seg_mask.reshape(128, 228, 1)).astype('uint8'), cv2.COLORMAP_HOT)
    heatmap_nao = cv2.applyColorMap((255*nao_mask.reshape(128, 228, 1)).astype('uint8'), cv2.COLORMAP_HOT)

    super_seg = cv2.addWeighted(heatmap_seg, 0.7, img, 0.3, 0)
    super_nao = cv2.addWeighted(heatmap_nao, 0.7, img, 0.3, 0)

    overall_image.paste(Image.fromarray(img), (0, 0))
    overall_image.paste(Image.fromarray(super_seg), (0, 128))
    overall_image.paste(Image.fromarray(super_nao), (0, 128*2))
    gif_list.append(overall_image)

    j += 1
    count += 1
    prev_start = start
    #cv2.imwrite('./results/seg_mask' + str(i) + '.png', 255*seg_mask)
    #cv2.imwrite('./results/gt_mask' + str(i) + '.png', 255*gt_mask)
    #cv2.imwrite('./results/image' + str(i) + '.png', sample_image[:,:,::-1])
