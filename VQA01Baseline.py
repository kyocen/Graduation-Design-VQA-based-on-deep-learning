import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as Dataloader
from torchvision import models, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import *

import os
import re
import glob
import pickle
import numpy as np
from PIL import Image
import sys

from VQA01DataProcess import *
from VQA01ImageProcess import *
from VQA01dataset import *


## The MLP baseline MLP层
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2048 + 300 + 300, 8192)
        self.fc2 = nn.Linear(8192, 1)

    def forward(self, x):
        # x = x.view(-1, 2048 + 300 + 300)
        x = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(x))
        return y


def loss_acc(loader, net, criterion, split):
    loss = 0.0
    correct = 0
    num_samples = 0

    # 一个sample是一个经过处理的iterator of batch，2d tensor，sample[i]是所有单个样本第i个的属性的集合：1d tensor，！！把他当做单个样本处理即可！！
    for i, sample in enumerate(loader, 1):  # sample是iterator of batch, dataloader中每一列的数据是tensor
        sample_var = [Variable(d).cuda() for d in list(sample)]
        X = sample_var[0]
        Y = sample_var[1]
        num = Y.data.size(0)
        num_samples += num

        y_hat = net(X)
        loss += criterion(y_hat, Y).data[0] * num  # 虽然是处理一个batch，但loss是一个scalar，是batch内所有样本的loss的平均，但是是一个tensor

        values, ids = torch.max(y_hat.data.view(-1, 18), 1)
        values_c, ids_c = torch.max(Y.data.view(-1, 18), 1)

        correct_batch = list(map(lambda x, y: 0 if x == y else 1, ids, ids_c)).count(0)
        correct += correct_batch
        if i % 900 == 0:  # print per 180 batches
            print('[%5d] running_loss(%s): %.3f accuracy(%s): %.3f' % (
                i, split, loss / num_samples, split, correct / num_samples * 18 * 100))

    return loss / num_samples, correct / num_samples * 18 * 100


BATCH_SIZE = 100 * 18

# test
if __name__ == '__main__':
    torch.cuda.set_device(0)  # 用0号gpu,注意torch与linux显示的gpu序号相反

    train_set = VQA01DataSet("train")
    eval_set = VQA01DataSet("eval")
    print("data set prepared!!!")
    # Data loader Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # If True, the data loader will copy tensors into CUDA pinned memory before returning them
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # train_samples.clear()
    # eval_samples.clear()
    # f_dict.clear()
    # emb_dict.clear()

    # build MLP model
    if torch.cuda.is_available():
        net = MLP().cuda()
        criterion = nn.BCELoss().cuda()
    else:
        net = MLP()
        criterion = nn.BCELoss()

    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=1e-3)
    running_loss = 0.0
    print("model prepared!!! train start!!!")

    for epoch in range(50):
        net.train()  # Sets the module in train mode.
        scheduler.step()
        running_loss = 0.0
        for i, sample in enumerate(train_loader, 1):
            # sample:[x,y] x:ndarray 2048+300+300d y:int 1/0
            # 一个sample是一个经过处理的batch，2d tensor，sample[i]是所有单个样本第i个的属性的集合：1d tensor，！！把他当做单个样本处理即可！！
            sample_var = [Variable(d).cuda() for d in list(sample)]
            ans = net.forward(sample_var[0])
            loss = criterion(ans, sample_var[1])  # tensor, 虽然是处理一个batch，但loss是一个scalar，是batch内所有样本的loss的平均，但是是一个tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tmp_loss = loss.data[0]
            running_loss += tmp_loss
            if i % 180 == 0:  # print every 180 batches
                print('[%5d] running_loss(train): %.3f' % (i, running_loss / i * 100))

        if epoch % 5 == 0:
            net.eval()  # Sets the module in evaluation mode. This has any effect only on modules such as Dropout or BatchNorm.
            train_loss, train_acc = loss_acc(train_loader, net, criterion, "train")
            eval_loss, eval_acc = loss_acc(eval_loader, net, criterion, "eval")
            print('[epoch: %d] train_loss: %.4f, valid_loss: %.4f, train_acc: %.4f, valid_acc: %.4f' % (epoch,
                                                                                                        train_loss,
                                                                                                        eval_loss,
                                                                                                        train_acc,
                                                                                                        eval_acc))
