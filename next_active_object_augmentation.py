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
import random
import cv2
from torchvision.transforms.functional import hflip
from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
     self.image_paths = image_paths
     self.target_paths = target_paths
     self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

     self.len = len(image_paths)

    def get_seg(self, mask):
        res_contact = np.where(mask < 9)
        res_no_contact = np.where(mask >= 9)
        mask[res_contact[0], res_contact[1]] = 0
        mask[res_no_contact[0], res_no_contact[1]] = 1
        return mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = image.resize((228, 128))
        image = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)(image)
        mask = np.load(self.target_paths[index])
        #print('unflipped', mask)
        if random.random() > 0.5:
            image = hflip(image)
            mask = np.array(hflip(Image.fromarray(mask)))
            #print('flipped', mask)
        seg_mask = self.get_seg(mask.copy())

        overall_mask = np.zeros((1, 128, 228))
        seg_mask = cv2.resize(seg_mask, (228, 128), interpolation=cv2.INTER_NEAREST)

        overall_mask[0] = seg_mask

        overall_mask = torch.from_numpy(overall_mask).type(torch.FloatTensor)
        image = torch.from_numpy(np.array(image.copy()).transpose(2, 0, 1)).type(torch.FloatTensor)
        image = self.normalize(image)

        #image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        #print(np.mean(image).cpu().numpy())
        return image, overall_mask

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
        #self.fc7 = ConvLSTMCell(4096, 4096, 1, 0)
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
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
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
conts = []
no_conts = []

def loss_seg_fn(output, target):
    weight = torch.tensor([1.0]).cuda()
    loss_fn = nn.BCELoss()
    loss = loss_fn(output.cuda(), target.cuda())
    return loss

def compare_contacts(output, target):
    res = np.where(np.array(target) == 0)
    loss = np.mean(np.abs(output[res] - target[res]))
    return loss

def compare_no_contacts(output, target):
    res = np.where(np.array(target) > 0)
    loss = np.mean(np.abs(output[res] - target[res]))
    return loss

def train_epoch(epoch, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()

    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))

        target = target.to(device)
        loss = loss_seg_fn(output.reshape(-1,).to(device), target.reshape(-1,).to(device))

        target_cont = target[:,0].data.cpu().numpy().reshape(-1)
        out_probs_cont = output[:,0].data.cpu().numpy().reshape(-1)

        sampled_cont = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)

        jaccard = jsc(target_cont, sampled_cont)

        #jaccard = 0
        loss.backward()
        optimizer.step()
        accs.append(loss.item())
        conts.append(jaccard)
        if batch_idx % 32 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tJac: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), np.mean(np.array(accs[-100:])), np.mean(np.array(conts[-100:]))))

def validate(test_loader, model, device, gamma=0.2):
    model.eval()
    losses = []
    jaccards = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data.to(device))
            target = target.to(device)
            loss = loss_seg_fn(output[:,0].reshape(-1,).to(device), target[:,0].reshape(-1,).to(device))
            losses.append(loss.item())
            target_cont = target[:,0].data.cpu().numpy().reshape(-1)
            out_probs_cont = output[:,0].data.cpu().numpy().reshape(-1)
            sampled_cont = (np.random.rand(len(out_probs_cont)) < out_probs_cont).astype(int)
            jaccard = jsc(target_cont, sampled_cont)
            jaccards.append(jaccard)
    return np.mean(np.array(losses)), np.mean(np.array(jaccards))

def main():
    num_classes = 1
    in_batch, inchannel, in_h, in_w = 16, 3, 224, 224
    image_data = sorted(glob.glob('./train/images/*'))
    mask_data = sorted(glob.glob('./train/masks/*'))
    device = torch.device("cuda")
    net = FCN8s(num_classes).to(device)

    #try:
    #    checkpoint = torch.load('./weights/weights/nao.pt')
    #    net.load_state_dict(checkpoint['model_state_dict'])
    #    s = checkpoint['epoch']
    #except Exception:
    #    s = 0
    s = 0
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    indices = list(range(len(image_data)))
    split = int(np.floor(0.9 * len(image_data)))
    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    image_val_data = sorted(glob.glob('./val/images/*'))
    mask_val_data = sorted(glob.glob('./val/masks/*'))

    train_dataset = CustomDataset(image_data, mask_data, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=16, num_workers=1)
    test_loader = torch.utils.data.DataLoader(train_dataset, sampler=test_sampler, batch_size=16, num_workers=1)

    best_loss = 100
    print('Training session -- Next Active Object Single RGB Frame')
    for epoch in range(s, 200):
        loss, jaccard = validate(test_loader, net, device)
        print('Validation:', loss, best_loss, jaccard)
        train_epoch(epoch, net, device, train_loader, optimizer)
        if loss < best_loss:
            print('Saving model -- epoch no. ', epoch)
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict()}, './weights/nao.pt')
            best_loss = loss

if __name__ == '__main__':
    main()
