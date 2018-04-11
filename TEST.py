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
from CSFMODEL import model

model.eval()# img: [bs,2048,7,7] que: (bs,14)
img=Variable(torch.randn(3,2048,7,7))
que=Variable(torch.arange(0,3.0*14.0).long().view(3,14))
print(que)
ouput=model.forward(que,img)
print(ouput.size())
print(ouput)



