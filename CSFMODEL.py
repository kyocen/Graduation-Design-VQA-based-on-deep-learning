import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as Dataloader
from torchvision import models, transforms
from torch.autograd import Variable
# from torch.optim.lr_scheduler import *

import os
import math
import re
import glob
import pickle
import numpy as np
from PIL import Image
import sys
import resnet
from modules import MFH, GatedTanh, CSF

stdModule = resnet.resnet152(True)
#print(list(list(stdModule.layer4.children())[0:-1]))


class CSFMODEL(nn.Module):
    def __init__(self, layer, num_words, num_ans, emb_size=300, inplanes=512 * 4, planes=512, stride=1):
        super(CSFMODEL, self).__init__()
        self.layers=layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)


        # 一开始的input是B x S, 但是Embedding S x B -> S x B x I，所以要先转置成S x B
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True)
        self.grudp = nn.Dropout(0.3)

        # CSF(img_size, h_size, latent_dim, output_size, block_count)  img_size=[C,H,W]
        self.csf1 = CSF((512, 7, 7), 512, 4, 1024, 2)
        self.csf2 = CSF((512, 7, 7), 512, 4, 1024, 2)
        self.csf3 = CSF((2048, 7, 7), 512, 4, 1024, 2)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(7, stride=1)
        self.fc = nn.Linear(512 * 4, 1024)

        self.pred_mfh = MFH(x_size=1024, y_size=512, latent_dim=4, output_size=1024,
                            block_count=2)  # (batch_size,36,o) or (batch_size,o)
        # self.pred_net = nn.Sequential(
        #     nn.Linear(2048, num_ans),
        #     nn.Sigmoid())
        self.pred_net=nn.Linear(2048, num_ans)

        # initialization
        # Returns an iterator over all modules in the network. Duplicate modules are returned only once.
        # 这一部分用来对模型中的conv随机初始化参数, 对batchnorm层初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # kernel: H*W*C
                # Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
                # normal_(mean=0, std=1, *, generator=None)
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        stdconv1 = list(stdModule.layer4.children())[2].conv1
        self.conv1.weight = nn.Parameter(list(stdconv1.parameters())[0].data.clone())

        stdbn1 = list(stdModule.layer4.children())[2].bn1
        self.bn1.weight = nn.Parameter(list(stdbn1.parameters())[0].data.clone())
        self.bn1.bias = nn.Parameter(list(stdbn1.parameters())[1].data.clone())
        if self.layers<3:
            for param in list(self.conv1.parameters()):
                param.requires_gard = False
            for param in list(self.bn1.parameters()):
                param.requires_gard=False

        stdconv2 = list(stdModule.layer4.children())[2].conv2
        self.conv2.weight = nn.Parameter(list(stdconv2.parameters())[0].data.clone())

        stdbn2 = list(stdModule.layer4.children())[2].bn2
        self.bn2.weight = nn.Parameter(list(stdbn2.parameters())[0].data.clone())
        self.bn2.bias = nn.Parameter(list(stdbn2.parameters())[1].data.clone())
        if self.layers<2:
            for param in list(self.conv2.parameters()):
                param.requires_gard = False
            for param in list(self.bn2.parameters()):
                param.requires_gard=False

        stdconv3 = list(stdModule.layer4.children())[2].conv3
        self.conv3.weight = nn.Parameter(list(stdconv3.parameters())[0].data.clone())

        stdbn3 = list(stdModule.layer4.children())[2].bn3
        self.bn3.weight = nn.Parameter(list(stdbn3.parameters())[0].data.clone())
        self.bn3.bias = nn.Parameter(list(stdbn3.parameters())[1].data.clone())
        if self.layers<1:
            for param in list(self.conv3.parameters()):
                param.requires_gard = False
            for param in list(self.bn3.parameters()):
                param.requires_gard=False

    def forward(self, que, img):  # img: [bs,2048,7,7] que: (bs,14)

        # process que
        # (bs,14) => (bs,14,300) question为14个word index, list 1d length 14, 每次forward都只对一个batch #2d tensor
        emb = F.tanh(self.we(que))
        # (bs, 14,300)->(1, bs, 512) question vector 只取最后的H (num_layers * num_directions, batch_size, hidden_size) 所以要squeeze(dim=0)
        _, h = self.gru(emb)
        h = self.grudp(h).squeeze(dim=0)  # (bs, 512)

        # process image tensor
        origin = img.clone()

        # first conv
        img = self.conv1(img)  # [bs,512,7,7]
        img = self.bn1(img)  # [bs,512,7,7]

        # first CSF
        # (bs,512,7,7) (bs,h_size) => (bs,512,7,7)
        if self.layers>=3 :
            img = self.csf1(img, h)

        # second conv
        img = self.conv2(img)  # [bs,512,7,7]
        img = self.bn2(img)  # [bs,512,7,7]

        # second CSF
        # (bs,512,7,7) (bs,h_size) => (bs,512,7,7)
        if self.layers>=2 :
            img = self.csf2(img, h)

        # third conv
        img = self.conv3(img)  # [bs,2048,7,7]
        img = self.bn3(img)  # [bs,2048,7,7]

        # third CSF
        # (bs,2048,7,7) (bs,h_size) => (bs,2048,7,7)
        if self.layers>=1 :
            img = self.csf3(img, h)

        img = img + origin  # (bs,2048,7,7)
        img = self.relu(img)  # (bs,2048,7,7)

        img_feature = self.maxpool(img)  # (bs,2048,1,1)
        img_feature = img_feature.view(img_feature.size(0), -1)  # (bs,2048)
        img_feature = self.fc(img_feature)  # (bs,1024)

        fuse = self.pred_mfh(img_feature, h)  # (bs,1024) (bs,512) => (bs,2048)
        score = self.pred_net(fuse)#(bs,3092)
        return score

