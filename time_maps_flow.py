import os
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import glob
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import jaccard_score as jsc
from torch.autograd import Variable
from skimage.transform import resize
import cv2
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms.functional import hflip
import random

class CustomDataset(Dataset):
    def __init__(self, image_paths, flow_paths, target_nao_paths, target_conts_paths, clip_length = 1, train=True):
     self.image_paths = image_paths
     self.target_nao_paths = target_nao_paths
     self.target_conts_paths = target_conts_paths
     self.flow_paths = flow_paths
     self.clip_length = clip_length
     self.len = len(image_paths)
     self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def get_contacts(self, mask):
        res_contact = np.where(mask == 1)
        res_no_contact = np.where(mask != 1)

        mask[res_contact[0], res_contact[1]] = 1
        mask[res_no_contact[0], res_no_contact[1]] = 0
        return mask

    def get_time(self, mask):
        res_nao = np.where(mask >= 9)
        res_no_nao = np.where(mask < 9)

        mask[res_no_nao[0], res_no_nao[1]] = -1
        mask[res_nao[0], res_nao[1]] = (mask[res_nao[0], res_nao[1]] - 10.0) / 30.0
        return mask.astype(np.float32)

    def __getitem__(self, index):
        beg = index-self.clip_length
        end = index+self.clip_length
        image_file = self.image_paths[index].split('_')
        flow_file = self.flow_paths[index].split('_')
        idx = int(image_file[3].strip('.png'))
        image = Image.open(self.image_paths[index]).resize((228, 128))

        mask_cont = np.load(self.target_conts_paths[index])
        mask_nao = np.load(self.target_nao_paths[index])

        contact_mask, time_mask = self.get_contacts(mask_cont.copy()), self.get_time(mask_nao.astype(np.float32).copy())

        overall_mask = np.zeros((2, 128, 228))
        time_mask = cv2.resize(time_mask, (228, 128), interpolation=cv2.INTER_LINEAR)
        contact_mask = cv2.resize(contact_mask, (228, 128), interpolation=cv2.INTER_NEAREST)

        overall_mask[0] = contact_mask
        overall_mask[1] = time_mask

        flip = random.random() > 0.5

        if flip:
            image = hflip(image)
            overall_mask = np.flip(overall_mask, axis=1)

        overall_mask = torch.from_numpy(overall_mask.copy()).type(torch.FloatTensor)

        overall_image = torch.from_numpy(np.array(image.copy()).transpose(2, 0, 1)).type(torch.FloatTensor)
        overall_image = self.normalize(overall_image)

        img_size = (228, 128)
        last_img_filename = 'h'

        prev_image = image

        for i in range(idx, idx - 2 * self.clip_length, -2):
            flow_file[3] = str(i) + '.npy'
            flow_filename = '_'.join(flow_file)

            try:
                flow = np.load(flow_filename)[0]
                last_flow_filename = flow_filename
            except Exception:
                flow = np.load(last_flow_filename)[0]

            if flip:
                flow = np.flip(flow, axis=1)

            flow = torch.from_numpy(flow.copy()).type(torch.FloatTensor)
            overall_image = torch.cat((overall_image, flow), 0)

            prev_image = image

    #    masks = [torch.from_numpy(masks).type(torch.FloatTensor)]
        return overall_image, overall_mask

    def __len__(self):
        return self.len

# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):

    pretrained_model = \
        os.path.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        #Flow layers
        self.conv_flow1_1 = nn.Conv2d(6, 64, 3, padding=100)
        self.relu_flow1_1 = nn.ReLU(inplace=True)
        self.conv_flow1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_flow1_2 = nn.ReLU(inplace=True)
        self.pool_flow1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        self.conv_flow2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_flow2_1 = nn.ReLU(inplace=True)
        self.conv_flow2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu_flow2_2 = nn.ReLU(inplace=True)
        self.pool_flow2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        self.conv_flow3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu_flow3_1 = nn.ReLU(inplace=True)
        self.conv_flow3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu_flow3_2 = nn.ReLU(inplace=True)
        self.conv_flow3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu_flow3_3 = nn.ReLU(inplace=True)
        self.pool_flow3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8


        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

    #    self._initialize_weights()

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, state=None):
        h = x[:,:3]
        f = x[:,3:]

        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))


        #Flow forward-pass here
        f = self.relu_flow1_1(self.conv_flow1_1(f))
        f = self.relu_flow1_2(self.conv_flow1_2(f))
        f = self.pool_flow1(f)

        f = self.relu_flow2_1(self.conv_flow2_1(f))
        f = self.relu_flow2_2(self.conv_flow2_2(f))
        f = self.pool_flow2(f)

        f = self.relu_flow3_1(self.conv_flow3_1(f))
        f = self.relu_flow3_2(self.conv_flow3_2(f))
        f = self.relu_flow3_3(self.conv_flow3_3(f))

        #Sum flow and RGB features (could concatenate instead)
        h = torch.add(h, f)
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        #h, _ = self.fc7(h, None)
        h = self.fc7(h) #This is the ConvLSTM Block

        h = self.relu7(h)
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        #print(h[:,1].shape)
        h[:,0] = F.sigmoid(h[:,0].clone())
        h[:,1] = F.relu(h[:,1].clone())
        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

accs = []
var_gt = []
var_out = []

def loss_seg_fn(output, target):
    weight = torch.tensor([1.0]).cuda()
    loss_fn = nn.BCELoss()
    loss = loss_fn(output.cuda(), target.cuda())
    return loss

def l1(output, target):
    res = (target > 0).nonzero()
    #print(len(target[res]))
    loss = torch.mean(torch.abs(output[res] - target[res]))
    return loss

def validate(test_loader, model, device, gamma=0.2):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data.to(device))
            target = target.to(device)

            loss_1 = loss_seg_fn(output[:,0].reshape(-1,).to(device), target[:,0].reshape(-1,).to(device))
            loss_2 = l1(output[:,1].reshape(-1,).to(device), target[:,1].reshape(-1,).to(device))

            if not np.isnan(loss_2.item()):
                loss = torch.add(loss_1, gamma*loss_2)
            else:
                loss = loss_1
            losses.append(loss.item())
    return np.mean(np.array(losses))

def train_epoch(epoch, model, device, data_loader, test_loader, optimizer, best_loss, gamma=0.2):
    model.train()
    pid = os.getpid()

    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))

        target = target.to(device)

        loss_1 = loss_seg_fn(output[:,0].reshape(-1,).to(device), target[:,0].reshape(-1,).to(device))
        loss_2 = l1(output[:,1].reshape(-1,).to(device), target[:,1].reshape(-1,).to(device))

        if not np.isnan(loss_2.item()):
            loss = torch.add(loss_1, gamma*loss_2)
        else:
            loss = loss_1

        loss.backward()
        optimizer.step()

        if not np.isnan(loss_2.item()):
            accs.append(loss.item())
            var_gt.append(loss_1.item())
            var_out.append(loss_2.item())

        if batch_idx % 32 == 0:
            val_loss = validate(test_loader, model, device)
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tContact: {:.6f} Time: {:.6f}\tVal Loss: {:.6f}\tBest Loss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), np.mean(np.array(accs[-100:])), np.mean(np.array(var_gt[-100:])),
                np.mean(np.array(var_out[-100:])), val_loss, best_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                print('Saving model -- epoch no. ', epoch)
                torch.save({'epoch': epoch, 'loss': best_loss, 'model_state_dict': model.state_dict()}, './weights/time_maps_flow.pt')
            model.train()
    return best_loss

def main():
    num_classes = 2
    clip_length = 3

    in_batch, inchannel, in_h, in_w = 16, 3, 224, 224
    image_data = sorted(glob.glob('./train/images/*'))
    flow_data = sorted(glob.glob('./train/flow/*'))
    mask_nao_data = sorted(glob.glob('./train/masks_nao/*'))
    mask_cont_data = sorted(glob.glob('./train/masks_cont/*'))
    device = torch.device("cuda")
    net = FCN8s(num_classes).to(device)

    try:
        checkpoint = torch.load('./weights/time_maps_flow.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        s = checkpoint['epoch']
        best_loss = checkpoint['loss']
    except Exception:
        s = 0
        best_loss = 100

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    image_val_data = sorted(glob.glob('./val/images/*'))
    flow_val_data = sorted(glob.glob('./val/flow/*'))
    mask_nao_val_data = sorted(glob.glob('./val/masks_nao/*'))
    mask_cont_val_data = sorted(glob.glob('./val/masks_cont/*'))

    train_dataset = CustomDataset(image_data, flow_data, mask_nao_data, mask_cont_data, clip_length=clip_length, train=True)
    val_dataset = CustomDataset(image_val_data, flow_val_data, mask_nao_val_data, mask_cont_val_data, clip_length=clip_length, train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=1)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=1)

    print('Training session -- Time Maps (Flow)')
    for epoch in range(s, 200):
        best_loss = train_epoch(epoch, net, device, train_loader, test_loader, optimizer, best_loss)


if __name__ == '__main__':
    main()
