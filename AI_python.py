import pickle
import itertools
import math

import torch
from torch import optim
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import numpy as np

def toYUV(rgb):
    rgb = rgb.numpy()
    R, G, B = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 *B
    U = -0.09991 * R + -0.33609 * G + 0.436 * G
    V = 0.615 * R + -0.55861 * G - 0.05639 * B
    return torch.from_numpy(np.asarray([Y, U, V]).reshape(3, 64, 64))
def toRGB(yuv, batchsize):
    """shape of yuv is bs x 3 x 64 x 64, ordered by YUV"""
    lst = []
    for data in yuv:
        Y, U, V = data[0, :, :], data[1, :, :], data[2, :, :]
        R = Y + 1.28033 * V
        G = Y - 0.21482 * U - 0.38059 * V
        B = Y + 2.12798 * U
        lst.append([R,G,B])
    return np.asarray(lst).reshape(batchsize, 3, 64, 64)#.clip(0, 255)

def make_file_list():
    train_img_list = list()
    for img_idx in range(14):
        img_path = "./dataset/"+ str(img_idx + 1) + ".jpeg"
        train_img_list.append(img_path)
    return train_img_list

class Img_Dataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed

class ImageTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((45,80)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: toYUV(x))
        ])

    def __call__(self, img):
        return self.data_transform(img)

train_dataset = Img_Dataset(file_list=make_file_list(),
                            transform=ImageTransform())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
batchSize = 1


def extractGray(batchSize, yuv):
    lst = []
    for data in yuv:
        lst.append(data[0])
    return np.asarray(lst).reshape(batchSize, 1, 64, 64)


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.cnn = nn.Sequential(
            # 3 x 80 x 45
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 40 x 22
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 20 x 11
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256 x 10 x 5
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            
            # 512 x 5 x 2
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        # input is real or fake colored image
        x = self.cnn(input)
        x = x.view(x.size(0), 512 * 4 * 4) # flatten it
        output = self.fc(x)
        return output.view(-1,1).squeeze(1)


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.fc = nn.Linear(100, 1 * 64 * 64)
        self.conv1 = nn.Conv2d(2, 126, 3, 1, 1, bias=False)#130
        self.bn1 = nn.BatchNorm2d(126)#130

        self.conv2 = nn.Conv2d(128, 62, 3, 1, 1, bias=False)#132
        self.bn2 = nn.BatchNorm2d(62)#66

        self.conv3 = nn.Conv2d(64, 31, 3, 1, 1, bias=False)#68
        self.pool = nn. MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(31)#65

        self.conv4 = nn.Conv2d(32, 15, 3, 1, 1, bias=False)#66
        self.bn4 = nn.BatchNorm2d(15)#65

        self.conv5 = nn.Conv2d(16, 7, 3, 1, 1, bias=False)#66
        self.bn5 = nn.BatchNorm2d(7)#33

        self.conv6 = nn.Conv2d(8, 2, 3, 1, 1, bias=False)#34
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input, noise_pure):
        # input is grayscale image(Y of YUV), noise is random sampled noise
        noise = self.fc(noise_pure)
        noise = noise.view(noise.size(0), 1, 64, 64)

        # 2 x 64 x 64
        x = self.conv1(torch.cat([input, noise], dim=1))
        x = self.bn1(x)
        x = self.relu(x)

        input2 = torch.cat([input, x ,noise], dim=1)
        # 132 x 64 x 64
        x = self.conv2(input2)
        x = self.bn2(x)
        x = self.relu(x)

        input3 = torch.cat([input, x, noise], dim=1)
        # 68 x 64 x 64
        x = self.conv3(input3)
        x = self.bn3(x)
        x = self.relu(x)

        input4 = torch.cat([input, x], dim=1)
        # 66 x 64 x 64
        x = self.conv4(input4)
        x = self.bn4(x)
        x = self.relu(x)

        input5 = torch.cat([input, x], dim=1)
        # 66 x 64 x 64
        x = self.conv5(input5)
        x = self.bn5(x)
        x = self.relu(x)

        input6 = torch.cat([input, x], dim=1)
        # 34 x 64 x 64
        x = self.conv6(input6)
        x = self.relu(x)
        x = torch.cat([input, x], dim=1)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
netG = _netG().to(device)
netG.apply(weights_init)
print(netG)

netD = _netD().to(device)
netD.apply(weights_init)
print(netD)


criterion = nn.BCELoss().to(device)

input = torch.FloatTensor(batchSize, 3, 45, 80).to(device)
noise = torch.FloatTensor(batchSize, 100).to(device)

label = torch.FloatTensor(batchSize).to(device)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=0.0002,betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002,betas=(0.5, 0.999))
result_dict= {}
loss_D, loss_G = [], []
outf = './result_image'


netG_PATH = './results/netG.pth'
netD_PATH = './results/netD.pth'

netG = _netG()
netG.load_state_dict(torch.load(netG_PATH))
netG.eval()
netG.to(device)

netD = _netD()
netD.load_state_dict(torch.load(netD_PATH))
netD.eval()
netD.to(device)

for i, data in enumerate(train_loader, 0):
    data = data.to(device)
    batchSize = len(data)
    gray = extractGray(batchSize, data.cpu().numpy())
    grayv = Variable(torch.from_numpy(gray)).to(device)
        #############
        # D!        #
        #############
    netD.zero_grad()
        ##############
        # real image #
        ##############
    input.resize_as_(data).copy_(data)
    label.resize_(len(data)).fill_(real_label)
    inputv = Variable(input).to(device)
    labelv = Variable(label).to(device)

    output = netD(inputv)
    D_x = output.data.mean()

        ##############
        # fake image #
        ##############
    noise.resize_(batchSize, 100).uniform_(0,1)
    noisev = Variable(noise).to(device)

        # create fake images
    fake = netG(grayv, noisev)
    fake.size()
        # cal loss
    output = netD(fake.detach())
    labelv = Variable(label.fill_(fake_label)).to(device)
    D_G_z1 = output.data.mean()
    
    optimizerD.step()

        ##############
        # G!         #
        ##############
    netG.zero_grad()
    labelv = Variable(label.fill_(real_label)).to(device)
    output = netD(fake)
        
    errG = criterion(output, labelv)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()

    
    rgb = toRGB(fake.cpu().data.numpy(), batchSize)
    real = toRGB(data.cpu().data.numpy(), batchSize)
    vutils.save_image(torch.from_numpy(rgb),'%s/fake_samples_epoch_%s.png' % (outf, i))