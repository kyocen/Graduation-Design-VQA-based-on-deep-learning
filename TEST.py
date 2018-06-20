import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models, transforms
from torch.autograd import Variable

import os
import re
import glob
import pickle
import json
import numpy as np
from PIL import Image
import sys
import argparse

from resnet import MyResNet,myresnet152
from CSFMODEL import CSFMODEL
from MFHMODEL import MFHMODEL
from MFHBaseline import MFHBaseline
from modules import MFH,CS,CSF

# model=MFHMODEL(layers=3,submodel='cs',grad=1,num_words=11896,num_ans=3098,hidden_size=1024)
# model=CSFMODEL(layers=3,submodel='cs',grad=1,num_words=11896,num_ans=3098,hidden_size=1024)
# model=MFHBaseline(layers=0,submodel='cs',grad=0,num_words=11896,num_ans=3097,hidden_size=1024,co_att=True)
# model=MFH([2048,7,7],y_size=1024,latent_dim=4,output_size=1024, block_count=2)
# model=CSF([2048,7,7],h_size=1024,latent_dim=4,output_size=1024, block_count=2)
# model=CS([2048,7,7],h_size=1024,latent_dim=4,)

# model.eval()# img: [bs,2048,7,7] que: (bs,14)
# img=Variable(torch.randn(3,2048,7,7))
# que=Variable(torch.arange(0,3.0*14.0).long().view(3,14))
# h=Variable(torch.randn(3,1024))
#
# ouput=model.forward(que,img)
# print(ouput.size())
# print(ouput)

# model=myresnet152(pretrained=True)
# ms=list(model.layer4.modules())
# print(len(ms))
# print(ms[0])
# print('##################################################################')
# print(ms[1])
# print('##################################################################')
# print(ms[2])

#model=CSFMODEL(3,'cs',0,11987,3000)
#model=CS((2048,7,7),1024)
model=MFH(1024,512,4,1024,2)
# for name,param in model.named_parameters():
#     print(name, param.size())
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
# print(list(model.parameters()))
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(list(model.named_modules()))