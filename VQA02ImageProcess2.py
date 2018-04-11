#处理VQA2.0图片集，提取feature并存入文件
#将每个batch的image feature存成dict_batchID.npy 保存一个dict: image_id->image feature
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

parser=argparse.ArgumentParser(description="VQA")
parser.add_argument("-gpu",type=int,action="store",help="gpu_index",default=3)
parser.add_argument("-bs",type=int,action="store",help="BATCH_SIZE",default=10)
args=parser.parse_args()


BATCH_SIZE=args.bs
my_module=myresnet152(pretrained=True)#提取模型


# feature save to disk 把图片转为feature 将每个image feature存成.npy 每个image feature存成一个文件
def get_image_feature(img_dir,f_dir,bs):
    if torch.cuda.is_available():
        image_module=my_module.cuda()
    else:
        image_module = my_module
    image_module.eval()
    # transforms.Compose(transforms)将多个操作组合成一个函数
    transform=transforms.Compose([
        transforms.ToTensor(),# range [0, 255] -> [0.0,1.0]转为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# 将图片各个维度normalize
    ])

    batch = []  # 一个batch中所有图片的集合
    image_ids = []  # 一个batch中所有image_id的集合
    batch_id = 1
    pattern=os.path.join(img_dir,"*.jpg")
    for i,filepath in enumerate(glob.glob(pattern),1):
        id=(os.path.basename(filepath).split('_')[-1]).split('.')[0]
        image_ids.append(id)
        img = Image.open(filepath).convert("RGB")# 将图片转为RBG模式,为啥要转呢
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = transform(img)# 转为tensor并normalize (3, 224, 224)
        img = img.unsqueeze(0)# 加一个维度变4维，为了之后cat (1, 3, 224, 224)
        batch.append(img)# batch为5维 [(1, 3, 224, 224)]

        if i%bs==0:
            batch=torch.cat(batch)  # batch变4维 (batchsize, 3, 224, 224)
            if torch.cuda.is_available():
                output=image_module(Variable(batch).cuda())
            else:
                output=image_module(Variable(batch))

            output=(output.data).cpu().numpy()#ndarray (bs,2048,7,7)
            for image_id, feature in zip(image_ids,output):
                np.save(os.path.join(f_dir,image_id),feature)# Save an 3d ndarray (2048,7,7) to a binary file in NumPy .npy format.

            batch_id+=1
            batch = []
            image_ids = []
            print("num of batch: ", i / bs)

    if len(batch)>0:
        batch = torch.cat(batch)  # batch变4维 (batchsize, 3, 224, 224)
        if torch.cuda.is_available():
            output = image_module(Variable(batch).cuda())
        else:
            output = image_module(Variable(batch))

        output = (output.data).cpu().numpy()  # ndarray (bs,2048,7,7)
        for image_id, feature in zip(image_ids, output):
            np.save(os.path.join(f_dir, image_id),feature)  # Save an 3d ndarray (2048,7,7) to a binary file in NumPy .npy format.

if __name__=="__main__":
    torch.cuda.set_device(args.gpu)
    #############################################################################
    # get offine images features
    get_image_feature("../data/vqa02/train2014","../data/vqa02/train2014_feature_2",BATCH_SIZE)
    get_image_feature("../data/vqa02/val2014","../data/vqa02/val2014_feature_2",BATCH_SIZE)
