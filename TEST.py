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

#model=MFHMODEL(layers=0,num_words=11896,num_ans=3098,hidden_size=1024)
#model=CSFMODEL(layers=3,num_words=11896,num_ans=3098,hidden_size=1024)
model=MFHBaseline(layers=0,num_words=11896,num_ans=3098,hidden_size=1024)

model.eval()# img: [bs,2048,7,7] que: (bs,14)
img=Variable(torch.randn(3,2048,7,7))
que=Variable(torch.arange(0,3.0*14.0).long().view(3,14))
print(que)
ouput=model.forward(que,img)
print(ouput.size())
print(ouput)



