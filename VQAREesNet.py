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
import numpy as np
from PIL import Image
import sys

from VQAv1Dataloader import *


## resnet101 (removed last layer) for feature extraction CNN除最后一层fc
class _resnet101(nn.Module):
    def __init__(self, origin_model):
        super(_resnet101, self).__init__()
        self.resnet_model = origin_model
        self.resnet_model.fc = nn.Dropout(1)

        for param in list(self.resnet_model.parameters())[:]:
            param.requires_gard = False

    def forward(self, x):
        y = self.resnet_model(x)
        return y


## The MLP baseline MLP层
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048 + 300 + 300, 8192)
        self.fc2 = nn.Linear(8192, 1)

    def forward(self, x):
        x = x.view(-1, 2048 + 300 + 300)
        x = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(x))
        return y


# read feature file 从图片的npy文件中提取dictionary，以文件名为key，以feature为value
def load_feature(path):
    dict = {}
    pattern = os.path.join(path, '*.npy')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        feature_batch = np.load(filepath)
        for key in feature_batch.item():
            dict[key] = feature_batch.item()[key]
    print(len(dict))

    return dict


# feature extractor - save to disk 把图片转为feature
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
        # 将图片各个维度normalize，为啥要normalize???????????????
    ])

    batch = []  # 一个batch中所有图片的集合
    filenames = []  # 一个batch中所有图片名字的集合
    dict = {}  # 一个batch中图片名对应feature
    batch_id = 1
    pattern = os.path.join(image_dict, '*jpg')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        filenames.append(os.path.basename(filepath))
        im = Image.open(filepath).convert('RGB')  # 将图片转为RBG模式,为啥要转呢？？？？？？？？？？？？？？？？？？？
        im = im.resize((224, 224), Image.ANTIALIAS)
        im = transform(im)  # 转为tensor并normalize
        im.unsqueeze_(0)  # 加一个维度变4维，为了之后cat
        batch.append(im)  # batch为5维

        if i % batchsize == 0:
            batch = torch.cat(batch)  # batch变4维
            if torch.cuda.is_available():
                output = my_resnet(Variable(batch).cuda())
            else:
                output = my_resnet(Variable(batch))

            output = (output.data).cpu().numpy()  # put variable on cpu, transform Variable into numpy array
            for filename, feature in zip(filenames, output):
                dict[filename] = feature
            np.save(os.path.join(f_dict, 'dict_' + str(batch_id)),
                    dict)  # Save an array to a binary file in NumPy .npy format.
            batch_id += 1

            batch = []
            filenames = []
            dict = {}
            print(i / 47300 * 100, '%')

    if len(batch) != 0:
        batch = torch.cat(batch)  # batch变4维
        if torch.cuda.is_available():
            output = my_resnet(Variable(batch).cuda())
        else:
            output = my_resnet(Variable(batch))

        output = (output.data).cpu().numpy()  # put variable on cpu, transform Variable into numpy array
        for filename, feature in zip(filenames, output):
            dict[filename] = feature
        np.save(os.path.join(f_dict, 'dict_' + str(batch_id)),
                dict)  # Save an array to a binary file in NumPy .npy format.

    print('image to feature done')


# calculate the mean of embedding for a given string
def embedding_mean(str, emb_dict):
    if(len(str)==0):
        print("string is empty")
        sys.exit(1)

    str = str.lower()
    words = re.findall(r"[:]|[^\w\s]|\w+", str)

    mean = []
    zero=[0]*300
    zero=np.asarray(zero)

    for word in words:
        if word in emb_dict:
            mean.append(emb_dict[word])
        else:
            mean.append(zero)

    mean = np.asarray(mean)

    return np.mean(mean, axis=0)


# prepare training data 把question vector，question vector，answer vector作为X，将label作为Y，打包成batchx，batchy，再打包成mini_batches
# sample(filename, question, answer, split, label)
#248,349 questions for training, 121, 512 for validation, and 244, 302 for testing
def prepare_minibatch(sample_set, f_dict, emb_dict, batchsize):
    minibatch = []
    batch_X = []
    batch_Y = []
    print("sample_set: ",np.asarray(sample_set).shape)
    for i, sample in enumerate(sample_set, 1):
        for j, single in enumerate(sample,1):
            imagename = os.path.basename(single['filename'])
            q = single['question']
            a = single['answer']
            f = f_dict[imagename]
            emb_q = embedding_mean(q, emb_dict)
            emb_a = embedding_mean(a, emb_dict)
            x = np.concatenate([f, emb_q, emb_a])
            y = [int(single['label'])]
            batch_X.append(x)
            batch_Y.append(y)

        if i % batchsize == 0:
            batch_X = np.asarray(batch_X)
            batch_Y = np.asarray(batch_Y)
            minibatch.append((batch_X, batch_Y))
            batch_X = []
            batch_Y = []

    if len(batch_X) != 0:
        batch_X = np.asarray(batch_X)
        batch_Y = np.asarray(batch_Y)
        minibatch.append((batch_X, batch_Y))

    return minibatch


def loss_acc(minibatch, net, criterion):
    loss = 0.0
    correct = 0
    num_samples = 0

    for i, batch in enumerate(minibatch, 1):
        X, Y = batch
        num = Y.shape[0]
        num_samples += num
        if torch.cuda.is_available():
            X = Variable(torch.from_numpy(X).float()).cuda()
            Y = Variable(torch.from_numpy(Y).float()).cuda()
        else:
            X = Variable(torch.from_numpy(X).float())
            Y = Variable(torch.from_numpy(Y).float())

        y_hat = net(X)
        loss += criterion(y_hat, Y).data[0] * num  # 为啥只用data[0], 而不是总和

        values, ids = torch.max(y_hat.data.view(-1, 18), 1)
        values_c, ids_c=torch.max(Y.data.view(-1, 18), 1)
        correct+=list(map(lambda x, y: 0 if x==y else 1, ids, ids_c)).count(0)

    return loss / num_samples, correct / num_samples * 18


BATCH_SIZE = 50

# test
if __name__ == '__main__':
    ## load json
    torch.cuda.set_device(1)#用0号gpu,注意torch与linux显示的gpu序号相反
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

    # ##############################################################################
    # # parse glove
    # # glove中每个word后面跟300个数字最为一个line，将所有embedding整理成字典
    # path = '../dataset/GloVe/vectors.6B.300d.txt'
    # emb_dict = parse_glove(path)
    # print('load embedding done', len(emb_dict))
    #
    # ###########################################################################
    # # get dictionary from all the question and answer, dictionary is a set
    # dictionary = get_dictionary(np.concatenate((np.asarray(train_samples), np.asarray(eval_samples)), axis=0))
    # print(len(dictionary))
    # print(dictionary)
    #
    ###########################################################################
    # new_dict = {}
    # for word in dictionary:
    #     if word in emb_dict:
    #         new_dict[word] = emb_dict[word]
    # with open('../dataset/GloVe/300d_VQAv1_dict.pkl', 'wb') as f:
    #     json.dump(new_dict, f)
    #     # pickle.dump(new_dict, f, pickle.HIGHEST_PROTOCOL)
    # print('write 300d_VQAv1_dict.pkl done', len(new_dict))
    #
    # #############################################################################
    # # get offine images features E:\workPlaceForPython\dataset\COCO\c
    # get_resnet101_feature('../dataset/COCO/train2014/', '../dataset/COCO/train2014_224_feature/', BATCH_SIZE)
    # get_resnet101_feature('../dataset/COCO/val2014/', '../dataset/COCO/val2014_224_feature/', BATCH_SIZE)

    emb_dict = {}  # word - embedding dictionary string->300d ndarray
    with open('../dataset/GloVe/300d_VQAv1_dict.pkl', 'rb') as f:
        emb_dict = json.load(f)
    print('load 300d_VQAv1_dict.pkl done', len(emb_dict))

    # load image feature
    f_dict = load_feature('../dataset/COCO/train2014_224_feature/')
    eval_f_dict = load_feature('../dataset/COCO/val2014_224_feature/')

    train_minibatch = prepare_minibatch(train_samples, f_dict, emb_dict, BATCH_SIZE)
    eval_minibatch = prepare_minibatch(eval_samples, eval_f_dict, emb_dict, BATCH_SIZE)

    train_samples.clear()
    eval_samples.clear()
    f_dict.clear()
    emb_dict.clear()

    # build MLP model
    if torch.cuda.is_available():
        net = Net().cuda()
        criterion = nn.BCELoss().cuda()
    else:
        net = Net()
        criterion = nn.BCELoss()

    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    running_loss = 0.0
    for i, batch in enumerate(train_minibatch, 1):
        batch_x, batch_y = batch
        if torch.cuda.is_available():
            X = Variable(torch.from_numpy(batch_x).float()).cuda()
            Y = Variable(torch.from_numpy(batch_y).float()).cuda()
        else:
            X = Variable(torch.from_numpy(batch_x).float())
            Y = Variable(torch.from_numpy(batch_y).float())

        optimizer.zero_grad()

        y_hat = net(X)
        loss = criterion(y_hat, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 1000 == 0:  # print every 2000 mini-batches
            print('[%5d] running_loss(train): %.3f' % (i, running_loss / 1000))
            running_loss = 0.0

    net.eval()  # Sets the module in evaluation mode. This has any effect only on modules such as Dropout or BatchNorm.
    train_loss, train_acc = loss_acc(train_minibatch, net, criterion)
    eval_loss, eval_acc = loss_acc(eval_minibatch, net, criterion)
    print('train_loss: %.4f, valid_loss: %.4f, train_acc: %.4f, valid_acc: %.4f' % (train_loss, eval_loss, train_acc, eval_acc))
    net.train()  # Sets the module in train mode.
