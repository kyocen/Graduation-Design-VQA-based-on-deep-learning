import json
import os
import re
import glob
import numpy as np

from PIL import Image

## construct sample
def construct_sample(filename, question, answer, split, label):
    sample = {}

    sample['i'] = filename
    sample['q'] = question
    sample['a'] = answer
    sample['split'] = split
    sample['label'] = label

    return sample


# 从data的json中提取出一个一个sample, sample(filename, question, answer, split, label)
# 其中相同的question的四个答案的sample连在一起，最后一个为正确答案
def construct_dataset(data):
    dataset = []

    for im in data['images']:
        split = im['split']
        filename = im['filename']

        for qa in im['qa_pairs']:
            question = qa['question']

            for a in qa['multiple_choices']:
                sample = construct_sample(filename, question, a, split, 0)
                dataset.append(sample)

            answer = qa['answer']
            sample = construct_sample(filename, question, answer, split, 1)
            dataset.append(sample)

    return dataset


## split dataset 将所有sample分为train，vaild，test
def split_dataset(dataset):
    train_set = []
    valid_set = []
    test_set  = []

    for sample in dataset:
        if sample['split'] == 'train':
            train_set.append(sample)
        elif sample['split'] == 'val':
            valid_set.append(sample)
        else:
            test_set.append(sample)

    return train_set, valid_set, test_set


## get dictionary 从所有question和answer中构造字典
def get_dictionary(dataset):
    dictionary = set()
    for sample in dataset:
        question = sample['q']
        add_to_dict(question, dictionary)
        answer = sample['a']
        add_to_dict(answer, dictionary)

    return dictionary


## add to dictionary 通过正则表达式从question和answer中提取单词加入字典
def add_to_dict(str, dictionary):
    str = str.lower()
    txt = re.findall(r"[:]|[^\w\s]|\w+", str)

    for word in txt:
        if word not in dictionary:
            dictionary.add(word)


## parse GloVe to dict glove中每个word后面跟300个数字最为一个line，将所有embedding整理成字典
def parse_glove(path):
    w2v = {}
    with open(path, encoding="utf-8",mode='r') as f:
        for line in f:
            words = line.split(' ')
            word = words[0]
            numbers = words[1:]
            l = []
            for num in numbers:
                l.append(float(num))
            w2v[word] = np.asarray(l)

    return w2v


## resize images - done once and for ever
def resize_images(im_size, src, dst):
    pattern = os.path.join(src, '*.jpg')
    #glob.glob()返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径
    for filepath in glob.glob(pattern):
        im = Image.open(filepath)
        im = im.resize((im_size, im_size), Image.ANTIALIAS)
        im.save(os.path.join(dst, os.path.basename(filepath)))


## Parameters
IM_SIZE = 224

if __name__ == '__main__':

    ## load json
    data=json.load(open('../dataset/Visual7W/dataset_v7w_telling/dataset_v7w_telling.json','r'))

    ## construct dataset from json
    dataset=construct_dataset(data)

    ## split dataset into train, valid and test
    train_set,vaild_set,test_set=split_dataset(dataset)
    print(len(train_set),len(vaild_set),len(test_set))

    ## get dictionary for all questions and answers
    dictionary=get_dictionary(dataset)
    print(len(dictionary))
    print(dictionary)

    ## parse GloVe
    path="../dataset/GloVe/vectors.6B.300d.txt"
    w2v = parse_glove(path)
    print(len(w2v))

    ## simple statistics统计voc中与question和answer中单词的交集
    cou=0
    for word in dictionary:
        if word not in w2v:
            print(word," not in w2v")
            cou+=1
    print(cou)
    print(len(dictionary))
    print(len(w2v))

    ## resize images
    # resize_images(IM_SIZE, '../dataset/Visual7W/images/', '../dataset/Visual7W/ResizeImages/')





