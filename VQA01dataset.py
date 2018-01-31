import json
import h5py
import pickle
import os
import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from VQA01DataProcess import *
from VQA01ImageProcess import *


class VQA01DataSet(Dataset):
    def __init__(self, split):
        ## load json
        # {'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans}
        if split == "train":
            self.data_set = json.load(open('../dataset/VQA/v1 Real Image/vqa_raw_train.json', 'r'))
        else:
            self.data_set = json.load(open('../dataset/VQA/v1 Real Image/vqa_raw_eval.json', 'r'))

        # construct dataset from json [{'ques_id': question_id(int), 'img_path': image_path(str), 'question': question(str), 'MC_ans': mc_ans(list of 18 str), 'ans': ans(str)}]
        # 从data的json中提取出一个一个sample, [[sample(filename(str), question(str), answer(str), split(str), label(bool))]] 248349*18 return list of list of dict
        # 其中相同的question的18个答案的sample连在一起
        # [[sample(filename(str), question(str), answer(str), split(str), label(bool))]]
        if split == "train":
            self.data_samples = self.construct_sample("train")
        else:
            self.data_samples = self.construct_sample("eval")
        print('load dataset done', self.data_samples.shape)

        self.emb_dict = {}  # word - embedding dictionary string->300d ndarray
        with open('../dataset/GloVe/300d_VQAv1_dict.pkl', 'rb') as f:
            self.emb_dict = pickle.load(f)
        print('load 300d_VQAv1_dict.pkl done', len(self.emb_dict))

        # load image feature, dict: image_name->image feature
        if split == "train":
            self.image_dict = self.load_feature('../dataset/COCO/train2014_224_feature/')
        else:
            self.image_dict = self.load_feature('../dataset/COCO/val2014_224_feature/')
        print('load image feature done', len(self.image_dict))

    def __getitem__(self, id):
        item = []
        # x = np.concatenate([f, emb_q, emb_a])
        # y = [int(single['label'])]
        row = int(id // 18)
        column = int(id % 18)
        sample = self.data_samples[row][column]
        f = self.image_dict[os.path.basename(sample['filename'])]
        q = self.embedding_mean(sample['question'])
        a = self.embedding_mean(sample['answer'])
        label = sample['label']
        x = np.concatenate([f, q, a]).astype(np.float32)
        y = np.asarray([label]).astype(np.float32)
        item.append(x)
        item.append(y)
        return item

    def __len__(self):
        return self.data_samples.shape[0]*self.data_samples.shape[1]

    # read feature file 从图片的npy文件中提取dictionary，以文件名为key，以feature为value, 所有batch和在一起, imagename->image feature
    def load_feature(self, path):
        dict = {}
        pattern = os.path.join(path, '*.npy')
        for i, filepath in enumerate(glob.glob(pattern), 1):
            feature_batch = np.load(filepath)
            for key in feature_batch.item():
                dict[key] = feature_batch.item()[key]
        print(len(dict))

        return dict

    # [{'ques_id': question_id(int), 'img_path': image_path(str), 'question': question(str), 'MC_ans': mc_ans(list of 18 str), 'ans': ans(str)}]
    # [[sample(filename(str), question(str), answer(str), split(str), label(bool))]] 248349*18 return list of list of dict
    def construct_sample(self, split):
        samples = []
        for x in self.data_set:
            onesample = []
            for i in range(18):
                sample = {}
                sample['filename'] = x['img_path']
                sample['question'] = x['question']
                sample['answer'] = x['MC_ans'][i]
                sample['split'] = split
                sample['label'] = int(sample['answer'] == x['ans'])
                onesample.append(sample)
            samples.append(onesample)
        samples=np.asarray(samples)
        print("samples: ", samples.shape)
        return samples

    # calculate the mean of embedding for a given string, return ndarray 300d float
    def embedding_mean(self, str):
        if (len(str) == 0):
            print("string is empty")
            sys.exit(1)

        str = str.lower()
        words = re.findall(r"[:]|[^\w\s]|\w+", str)

        mean = []
        zero = [0] * 300
        zero = np.asarray(zero)

        for word in words:
            if word in self.emb_dict:
                mean.append(self.emb_dict[word])
            else:
                mean.append(zero)

        mean = np.asarray(mean)

        return np.mean(mean, axis=0)
