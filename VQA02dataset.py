import json
import h5py
import os
import re
import glob
import pickle
import json

import numpy as np
import progressbar
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
import torch.utils.data

from config import cfg, get_feature_path

# # Save
# codebook = {}
# codebook['itoa'] = itoa
# codebook['itow'] = itow
# codebook_fname = '../data/vqa02/codebook.json'
# print('[Store] {}'.format(codebook_fname))
# json.dump(codebook, open(codebook_fname, 'w'))
#
# with h5py.File(train_h5, 'w') as f:
#     group = f.create_group('train')
#     group.create_dataset('que_id', dtype='int64', data=train_que_id)
#     group.create_dataset('que', dtype='int64', data=train_que)
#
#     group.create_dataset('ans', dtype='float16', data=train_ans)
#     group.create_dataset('correct', dtype='int64', data=train_correct)
#     group.create_dataset('img_id', dtype='int64', data=train_img_id)
#     # train_que中的每一行的question
#     # 对应 train_que_id中同一行的question_id
#     # 对应 train_ans中同一行的的答案分布
#     # 对应 train_correct中的correct answer index
#     # 对应 train_image_id中同一行的image_id
#All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset,
# and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
class VQA02Dataset(Dataset):
    def __init__(self,split,extend=1.0,freq=0):# split in {'train2014', 'val2014'}
        print('[Load] raw data for {}'.format(split))
        self.extend=float(extend)
        self.freq=freq
        with open('../data/vqa02/codebook.json', 'r') as f:
            self.codebook=json.load(f)
        self.img_feature_path='../data/vqa02/{}_feature_2'.format(split)
        with h5py.File('../data/vqa02/{}-paired.h5'.format(split)) as f:
            paired_data = f[split[0:-4]]
            # np.arrray 1维 长774931 array([458752000, 458752001, 458752002, ...,    955086,955088,955097])
            self.que_id = paired_data['que_id'].value
            self.img_id = paired_data['img_id'].value
            self.que = paired_data['que'].value  # ndarray 2维 (774931, 14)
            # array([[  0,   0,   0, ...,  24, 103, 436],
            #        [  0,   0,   0, ...,   9,  15,  80],
            #        [  0,   0,   0, ...,   1, 260,  42],
            #        ...,
            #        [  0,   0,   0, ...,   2,   1,  47],
            #        [  0,   0,   0, ...,   9, 122,  24],
            #        [  0,   0,   0, ..., 118,  62,   6]])
            self.ans = paired_data['ans'].value.astype(np.float32) # ndarray 2维 (774931, 3097)
            # print("que_id ",self.que_id.shape[0])
            # print("img_id ",self.img_id.shape[0])
            # print("que ",self.que.shape[0])
            # print("ans ",self.ans.shape[0])
            # print("correct_index ",self.correct_index.shape[0])
        self.img_feature = h5py.File('../data/vqa02/{}_img_feature.h5'.format(split))
        with open('../data/vqa02/{}_imgid_index.json'.format(split), 'r') as load_f:
            self.imgid_index=json.load(load_f)


    def load_feature(self,dir):
        print('[Load] image feature in {}'.format(dir))
        feature_dict={}
        bar = progressbar.ProgressBar()
        pattern=os.path.join(dir,"*.npy")
        for i,filepath in enumerate(bar(glob.glob(pattern)),1):
            feature_batch=np.load(filepath).item()
            for key in feature_batch:
                feature_dict[key]=feature_batch[key]
        print(len(feature_dict))
        return feature_dict

    def __len__(self):
        return self.que_id.shape[0]

    # [question [index of word] ndarray 1d, image feature  3d ndarray (2048,7,7),
    # 1d ndarray [score(float32) of N candidate answers for this question] ]
    def __getitem__(self, i):
        item=[]
        item.append(self.que_id[i]) #question id int64
        item.append(self.que[i])#[index of word] ndarray 1d
        imgid=str(self.img_id[i])
        img_index= self.imgid_index[imgid]
        img_feature = self.img_feature['img'][img_index]
        #print('[feature] {} :img_index {}'.format(i,img_index))
        item.append(img_feature)#image feature  3d ndarray (2048,7,7)
        #del img_feature
        item.append(self.ans[i])#1d ndarray [score(float32) of N candidate answers for this question]
        return item

    @property
    def num_words(self):  # 只读属性
        return len(self.codebook['itow'])

    @property
    def num_ans(self):  # 只读属性
        return len(self.codebook['itoa'])
#
# BATCH_SIZE=10
# train_set = VQA02Dataset('train2014')
# # Data loader Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
# train_loader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True,
#     )  # If True, the data loader will copy tensors into CUDA pinned memory before returning them
#
# dataiter=iter(train_set)
# item=next(dataiter)
# print("[size] size of item: ",len(item))
# print("[size] size of question: ",item[0].shape)
# print("[size] size of image feature: ",item[1].shape)
# print("[size] size of answer: ",item[2].shape)

#
# val_set = VQA02Dataset('val2014')
# val_loader = torch.utils.data.DataLoader(
#     val_set,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True,
# )






