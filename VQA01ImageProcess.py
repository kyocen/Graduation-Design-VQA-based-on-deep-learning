#提取feature和dictionary
#将每个batch的image feature存成dict_batchID.npy 保存一个dict: image_name->image feature
#将所有question和answer中提到的word存成dict: word->embedding
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

from VQA01DataProcess import *

## resnet101 (removed last layer) for feature extraction CNN除最后一层fc
class _resnet101(nn.Module):
    def __init__(self, origin_model):
        super(_resnet101, self).__init__()
        self.resnet_model = origin_model
        self.resnet_model.fc = nn.Dropout(1)

        for param in self.resnet_model.parameters():
            param.requires_gard = False

    def forward(self, x):
        y = self.resnet_model(x)
        return y


# feature extractor - save to disk 把图片转为feature 将每个batch的image feature存成dict_batchID.npy 保存一个dict: image_name->image feature
def get_resnet101_feature(image_dict, f_dict, batchsize):
    if not os.path.exists(f_dict):
        os.makedirs(f_dict)

    if torch.cuda.is_available():
        my_resnet = _resnet101(models.resnet101(pretrained=True)).cuda()
    else:
        my_resnet = _resnet101(models.resnet101(pretrained=True))

    # transforms.Compose(transforms)将多个操作组合成一个函数
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]转为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 将图片各个维度normalize
    ])

    batch = []  # 一个batch中所有图片的集合
    filenames = []  # 一个batch中所有图片名字的集合
    dict = {}  # 一个batch中图片名对应feature
    batch_id = 1
    pattern = os.path.join(image_dict, '*.jpg')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        filenames.append(os.path.basename(filepath))
        im = Image.open(filepath).convert('RGB')  # 将图片转为RBG模式,为啥要转呢？？？？？？？？？？？？？？？？？？？
        im = im.resize((224, 224), Image.ANTIALIAS)
        im = transform(im)  # 转为tensor并normalize (224, 224,3)
        im.unsqueeze_(0)  # 加一个维度变4维，为了之后cat (1, 224, 224,3)
        batch.append(im)  # batch为5维 [(1, 224, 224,3)]

        if i % batchsize == 0:
            batch = torch.cat(batch)  # batch变4维 (batchsize, 224, 224,3)
            if torch.cuda.is_available():
                output = my_resnet(Variable(batch).cuda())
            else:
                output = my_resnet(Variable(batch))

            output = (output.data).cpu().numpy()  # put variable.tensor on cpu, transform tensor into numpy array (batchsize,2048)
            for filename, feature in zip(filenames, output):#(bs) (bs)
                dict[filename] = feature
            np.save(os.path.join(f_dict, 'dict_' + str(batch_id)),dict)  # Save an array to a binary file in NumPy .npy format.
            batch_id += 1

            batch = []
            filenames = []
            dict = {}
            print("num of batch: ", i / batchsize)

    if len(batch) != 0:
        batch = torch.cat(batch)  # batch变4维 (batchsize, 224, 224,3)
        if torch.cuda.is_available():
            output = my_resnet(Variable(batch).cuda())
        else:
            output = my_resnet(Variable(batch))

        output = (output.data).cpu().numpy()  # put variable on cpu, transform Variable into numpy array (batchsize,2048)
        for filename, feature in zip(filenames, output):
            dict[filename] = feature
        np.save(os.path.join(f_dict, 'dict_' + str(batch_id)),dict)  # Save an array to a binary file in NumPy .npy format.

    print('image to feature done')


BATCH_SIZE = 50

# test
if __name__ == '__main__':#提取feature和dictionary
    torch.cuda.set_device(0)  # 用0号gpu,注意torch与linux显示的gpu序号相反
    ## load json
    #{'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans}
    train_set = json.load(open('../dataset/VQA/v1 Real Image/vqa_raw_train.json', 'r'))
    eval_set = json.load(open('../dataset/VQA/v1 Real Image/vqa_raw_eval.json', 'r'))

    # construct dataset from json [{'ques_id': question_id(int), 'img_path': image_path(str), 'question': question(str), 'MC_ans': mc_ans(list of 18 str), 'ans': ans(str)}]
    # 从data的json中提取出一个一个sample, [[sample(filename(str), question(str), answer(str), split(str), label(bool))]] 248349*18 return list of list of dict
    # 其中相同的question的18个答案的sample连在一起
    # [[sample(filename(str), question(str), answer(str), split(str), label(bool))]]
    train_samples = construct_sample(train_set, "train")
    eval_samples = construct_sample(eval_set, "eval")

    print('load dataset done', len(train_samples), len(eval_samples))

    ##############################################################################
    # parse glove
    # glove中每个word后面跟300个数字最为一个line，将所有embedding整理成字典
    path = '../dataset/GloVe/vectors.6B.300d.txt'
    emb_dict = parse_glove(path)# parse GloVe to dict, glove中每个word后面跟300个数字最为一个line，将所有embedding整理成字典, word -> ndarray of float 300d
    print('load embedding done', len(emb_dict))

    ###########################################################################
    # get dictionary from all the question and answer, dictionary is a set
    dictionary = get_dictionary(np.concatenate((np.asarray(train_samples), np.asarray(eval_samples)), axis=0))
    print(len(dictionary))

    ###########################################################################
    new_dict = {}
    for word in dictionary:
        if word in emb_dict:
            new_dict[word] = emb_dict[word]
    with open('../dataset/GloVe/300d_VQAv1_dict.pkl', 'wb') as f:
        #json.dump(new_dict,f)
        pickle.dump(new_dict, f, pickle.HIGHEST_PROTOCOL)
    print('write 300d_VQAv1_dict.pkl done', len(new_dict))

    # new_dict.clear()
    # dictionary.clear()
    # emb_dict.clear()
    # train_set.clear()
    # eval_set.clear()
    # train_samples.clear()
    # eval_samples.clear()

    #############################################################################
    # get offine images features E:\workPlaceForPython\dataset\COCO\
    get_resnet101_feature('../dataset/COCO/train2014/', '../dataset/COCO/train2014_224_feature/', BATCH_SIZE)
    get_resnet101_feature('../dataset/COCO/val2014/', '../dataset/COCO/val2014_224_feature/', BATCH_SIZE)


