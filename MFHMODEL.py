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
from config import cfg
stdModule = resnet.resnet152(True)
#print(list(list(stdModule.layer4.children())[0:-1]))


class MFHMODEL(nn.Module):
    def __init__(self, layers, num_words, num_ans, hidden_size=1024, emb_size=300, co_att=False, inplanes=512 * 4, planes=512, stride=1):
        super(MFHMODEL, self).__init__()
        self.layers=layers
        self.co_att=co_att
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 1024)

        # 一开始的input是B x S, 但是Embedding S x B -> S x B x I，所以要先转置成S x B
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        self.lstmdp1 = nn.Dropout(0.3)
        self.lstmdp2 = nn.Dropout(0.3)

        #question attention layer
        self.Conv1_Qatt = nn.Conv2d(1024, 512, 1)
        self.Conv2_Qatt = nn.Conv2d(512, cfg.NUM_QUESTION_GLIMPSE, 1)

        # CSF(img_size, h_size, latent_dim, output_size, block_count)  img_size=[C,H,W]
        self.csf1 = CSF((512, 7, 7), cfg.NUM_QUESTION_GLIMPSE*hidden_size, 4, 1024, 2)
        self.csf2 = CSF((512, 7, 7), cfg.NUM_QUESTION_GLIMPSE*hidden_size, 4, 1024, 2)
        self.csf3 = CSF((2048, 7, 7), cfg.NUM_QUESTION_GLIMPSE*hidden_size, 4, 1024, 2)


        self.att_mfh = MFH(2048, cfg.NUM_QUESTION_GLIMPSE*hidden_size, latent_dim=4, output_size=1024, block_count=2)#(batch_size,36,o) or (batch_size,o)
        self.att_net = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh = MFH(x_size=2048, y_size=cfg.NUM_QUESTION_GLIMPSE*hidden_size, latent_dim=4, output_size=1024,
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

        #resnet最后三层的初始化。并决定要不要继续训练这3层
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
        #qoutput  (batch, seq_len, hidden_size * num_directions)=(bs,14,1024)
        #h,c=hn hn:(2, num_layers * num_directions, batch, hidden_size) = (2,1,bs,1024)
        qouput, hn = self.lstm(emb)
        h,c=hn#(1, bs, 1024)

        if not self.co_att:
            h = self.lstmdp1(h).squeeze(dim=0)  # (bs, 1024)
        else:
            # question attention
            qouput=self.lstmdp2(qouput)  # (bs,14,1024)
            qouput = qouput.permute(0, 2, 1).unsqueeze(3)  # (bs,14,1024) => (bs,1024,14) => (bs,1024,14,1)
            qatt_conv1 = self.Conv1_Qatt(qouput)  # (bs,1024,14,1) => (bs,512,14,1)
            qatt_relu = F.relu(qatt_conv1)  # (bs,512,14,1)
            qatt_conv2 = self.Conv2_Qatt(qatt_relu)  # (bs,NUM_QUESTION_GLIMPSE,14,1)
            qatt_conv2 = F.softmax(qatt_conv2, dim=2)  # 对权重做softmax (bs,NUM_QUESTION_GLIMPSE,14,1)
            qatt_conv2 = qatt_conv2.squeeze(dim=3)  # (bs,NUM_QUESTION_GLIMPSE,14)
            qouput = qouput.squeeze(3).permute(0, 2, 1)  # (bs,1024,14,1) => (bs,1024,14) => (bs,14,1024)
            qatt_feature_list = []  # (NUM_QUESTION_GLIMPSE, bs, 1024)
            for i in range(cfg.NUM_QUESTION_GLIMPSE):
                t_qatt_mask = qatt_conv2.narrow(1, i, 1)  # N x 1 x 14
                q_att = torch.bmm(t_qatt_mask, qouput).squeeze(1)  # (bs, 1, 14) * (bs,14,1024) => (bs,1,1024) => (bs,1024)
                qatt_feature_list.append(q_att)
            h = torch.cat(qatt_feature_list, 1)  # (bs,1024*NUM_QUESTION_GLIMPSE)


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

        img=img.view(img.size(0),img.size(1),-1).permute(0,2,1)# (bs,2048,7,7) => (bs,2048,49) => (bs,49,2048)
        img_norm = F.normalize(img, p=2,dim=2)  # img(bs,49,2048),这里是feature vector内部normalize

        #image feature and attention结合
        # (batch_size,49,2048) (batch_size,512)->(batch_size,49,2048)->(batch_size,49,1)
        att_w = self.att_net(self.att_mfh(img_norm, h))  # (batch_size,49,o) => (batch_size,49,1)
        # (batch_size,36,1)->(36,batch_size,1)在36个box_weigh中做softmax normalize->(batch_size,1,36)
        att_w_exp = F.softmax(att_w,dim=1).permute(0,2,1)  # (batch_size,49,1) => (bs,1,49)

        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2. (batch_size,1,36)*(batch_size,36,2048)
        # batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        # If batch1 is a (b×n×m)tensor, batch2 is a (b×m×p)tensor, out will be a (b×n×p)tensor.
        img_feature = torch.bmm(att_w_exp,img_norm)  # att_w_exp(batch_size,1,49) img_norm(batch_size,49,2048) ->(batch_size,1,2048) 加权和


        # (batch_size,1,2048)
        img_feature = img_feature.squeeze(1)  # (bs,2048)

        #image feature and question feature 结合
        fuse = self.pred_mfh(img_feature, h)  # (bs,2048) (bs,1024) => (bs,2048)
        score = self.pred_net(fuse)#(bs,3098)
        score=F.softmax(score,dim=1)
        return score


