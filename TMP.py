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

from DataLoader import *


## VGG16 (removed last layer) for feature extraction CNN除最后一层softmax
class _VGG16(nn.Module):
    def __init__(self, original_model):
        super(_VGG16, self).__init__()
        self.features = original_model.features
        new_classifier  = nn.Sequential(*list(original_model.classifier.children())[:-1])
        self.classifier = new_classifier

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


## The MLP baseline MLP层
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4096+300+300, 8192)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8192, 1)

    def forward(self, x):
        x = x.view(-1, 4096+300+300)
        x = F.relu(self.dropout1(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(x))
        return y


## read feature file 从图片的npy文件中提取dictionary，以文件名为key，以feature为value
def load_features(path):
    dict = {}
    pattern = os.path.join(path, '*.npy')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        raw_array = np.load(filepath)
        for key in raw_array.item().keys():
            value = raw_array.item().get(key)
            dict[key] = value
    print (len(dict))

    return dict

## prepare offline data
def prepare_data(dataset, batchsize):
    for i, sample in enumerate(dataset):
        question = sample['q']
        answer = sample['a']
        im_file = sample['i']

        print (question, answer, im_file)

## feature extractor - save to disk 把图片转为feature
def get_vgg16_features(im_src, f_dst, batchsize):


    if not os.path.exists(f_dst):
        os.makedirs(f_dst)

    if torch.cuda.is_available():
        my_vgg = _VGG16(models.vgg16(pretrained=True)).cuda()
    else:
        my_vgg = _VGG16(models.vgg16(pretrained=True))
    #transforms.Compose(transforms)将多个操作组合成一个函数
    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]转为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#将图片各个维度normalize，为啥要normalize
    ])

    batch = []#一个batch中所有图片的集合
    file_names = []#一个batch中所有图片名字的集合
    dict = {}#一个batch中图片名对应feature
    batch_id = 1
    pattern = os.path.join(im_src, '*.jpg')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        file_name = os.path.basename(filepath)
        file_names.append(file_name)
        im = Image.open(filepath).convert('RGB')#将图片转为RBG模式,为啥要转呢？？？？？？？？？？？？？？？？？？？
        im = transform(im)#转为tensor并normalize
        im.unsqueeze_(0)#加一个维度变4维，为了之后cat
        batch.append(im)#batch为5维
        if i % batchsize == 0:
            batch = torch.cat(batch)#batch变4维
            if torch.cuda.is_available():
                outputs = my_vgg(Variable(batch).cuda())
            else:
                outputs = my_vgg(Variable(batch))
            outputs = (outputs.data).cpu().numpy()#put variable on cpu, transform Variable into numpy array
            for file_name, feature in zip(file_names, outputs):
                dict[file_name] = feature
            np.save(os.path.join(f_dst, 'dict_'+str(batch_id)), dict)#Save an array to a binary file in NumPy .npy format.
            batch_id += 1

            batch = []
            file_names = []
            dict = {}
            print (i / 47300 * 100, '%')

    if len(batch) != 0:
        batch = torch.cat(batch)
        if torch.cuda.is_available():
            outputs = my_vgg(Variable(batch).cuda())
        else:
            outputs = my_vgg(Variable(batch))
        outputs = (outputs.data).cpu().numpy()
        for file_name, feature in zip(file_names, outputs):
            dict[file_name] = feature
        np.save(os.path.join(f_dst, 'dict_'+str(batch_id)), dict)

    print ('Done.')


## calculate the mean of embedding for a given string
def get_emb_mean(sentence, emb_dict):
    sentence = sentence.lower()
    sentence = re.findall(r"[:]|[^\w\s]|\w+", sentence)

    embs = []
    for word in sentence:
        if word in emb_dict:
            embs.append(emb_dict[word])

    embs = np.asarray(embs)
    emb_mean = np.mean(embs, axis=0)

    return emb_mean


## prepare training data 把question vector，question vector，answer vector作为X，将label作为Y，打包成batchx，batchy，再打包成mini_batches
def prepare_mini_batch(dataset, im_dict, emb_dict, batch_size):
    mini_batches = []
    batch_X = []
    batch_y = []
    for seq, sample in enumerate(dataset, 1):
        i = sample['i']
        q = sample['q']
        a = sample['a']

        fea_i = im_dict[i]
        emb_q = get_emb_mean(q, emb_dict)
        emb_a = get_emb_mean(a, emb_dict)

        x = np.concatenate([fea_i, emb_q, emb_a])
        y = int(sample['label'])

        batch_X.append(x)
        batch_y.append([y])

        if seq % batch_size == 0:
            batch_X = np.asarray(batch_X)
            batch_y = np.asarray(batch_y)
            mini_batches.append((batch_X, batch_y))
            batch_X = []
            batch_y = []

    if len(batch_X) != 0:
        batch_X = np.asarray(batch_X)
        batch_y = np.asarray(batch_y)
        mini_batches.append((batch_X, batch_y))

    return mini_batches

def get_loss_and_acc(mini_batches, net, criterion):#evaluate
    loss = 0.0
    correct = 0
    n_samples = 0
    for i, mini_batch in enumerate(mini_batches, 1):
        X, y = mini_batch
        m = y.shape[0]
        n_samples += m
        if torch.cuda.is_available():
            X = Variable(torch.from_numpy(X).float()).cuda()
            y = Variable(torch.from_numpy(y).float()).cuda()
        else:
            X = Variable(torch.from_numpy(X).float())
            y = Variable(torch.from_numpy(y).float())
        y_hat = net(X)
        values, indices = torch.max(y_hat.data.view(-1, 4) ,1)#共有4个答案，最后一个为正确答案，将model给出的答案index vector转为0/1 vector
        indices = indices[indices == 3]#indices == 3返回一个index所在的值都为3的index vector
        if indices.size() != torch.Tensor([]).size():
            correct += indices.size()[0]
        loss += criterion(y_hat, y).data[0] * m # 为啥只用data[0], 而不是总和
    return loss / n_samples, correct / (n_samples/ 4.0)

BATCH_SIZE = 32

## tests
if __name__ == '__main__':
    ## load json
    data = json.load(open('../dataset.json', 'r'))

    ## construct dataset from json
    dataset = construct_dataset(data)

    ## split dataset into train, valid and test
    train_set, valid_set, test_set = split_dataset(dataset)
    print ('load dataset done', len(train_set), len(valid_set), len(test_set))

    # ## get dictionary for all questions and answers
    # dictionary = get_dictionary(dataset)
    # print (len(dictionary))
    # print (dictionary)

    ## parse GloVe
    path = '../glove/glove.42B.300d.txt'
    emb_dict = parse_glove(path)
    print ('load embedding done', len(emb_dict))

    # new_dict = {}
    # for word in dictionary:
    #    if word in emb_dict:
    #        new_dict[word] = emb_dict[word]
    # with open('../glove/reduced_dict.pkl', 'wb') as f:
    #    pickle.dump(new_dict, f, pickle.HIGHEST_PROTOCOL)
    # print ('Get reduced_dict done.', len(new_dict))

    emb_dict = {}# word - embedding dictionary
    with open('../glove/reduced_dict.pkl', 'rb') as f:
        emb_dict = pickle.load(f)
    print ('Load reduced emb_dict done.', len(emb_dict))

    # ## get offine images features
    # get_vgg16_features('../images/resize', '../features', BATCH_SIZE)

    ## load offline images features
    im_dict = load_features('../features/')

    train_mini_batches = prepare_mini_batch(train_set, im_dict, emb_dict, BATCH_SIZE)
    valid_mini_batches = prepare_mini_batch(valid_set, im_dict, emb_dict, BATCH_SIZE)

    ## build MLP model
    if torch.cuda.is_available():
        net = Net().cuda()
        criterion = nn.BCELoss().cuda()
    else:
        net = Net()
        criterion = nn.BCELoss()

    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    n_epochs = 500
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_mini_batches, 1):
            batch_X, batch_y = batch
            if torch.cuda.is_available():
                X = Variable(torch.from_numpy(batch_X).float()).cuda()
                y = Variable(torch.from_numpy(batch_y).float()).cuda()
            else:
                X = Variable(torch.from_numpy(batch_X).float())
                y = Variable(torch.from_numpy(batch_y).float())

            optimizer.zero_grad()

            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 1000 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] running_loss(train): %.3f' % (epoch + 1, i, running_loss / 1000))
                running_loss = 0.0

        net.eval()# Sets the module in evaluation mode. This has any effect only on modules such as Dropout or BatchNorm.
        train_loss, train_acc = get_loss_and_acc(train_mini_batches, net, criterion)
        valid_loss, valid_acc = get_loss_and_acc(valid_mini_batches, net, criterion)
        #y_hat_train = net(X_train_)
        #train_loss = criterion(y_hat_train, y_train_)
        #y_hat_valid = net(X_valid_)
        #valid_loss = criterion(y_hat_valid, y_valid_)
        print ('[%d] train_loss: %.4f, valid_loss: %.4f, train_acc: %.4f, valid_acc: %.4f' % (epoch+1, train_loss, valid_loss, train_acc, valid_acc))
        net.train()# Sets the module in train mode.
