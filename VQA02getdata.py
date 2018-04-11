import json
import argparse
from collections import Counter
from operator import itemgetter

import h5py
import numpy as np
import random

from config import cfg, get_feature_path

# trn_data list of dict train2014-raw.json 例子：
# [{"image_id": 458752, "question": ["what", "is", "this", "photo", "taken", "looking", "through"],
# "question_id": 458752000, "answers": [["net", 1.0], ["netting", 0.3], ["mesh", 0.3]]},...]

# val2014-raw.json 例子：
# {"image_id": 262148, "question": ["where", "is", "he", "looking"],
# "question_id": 262148000, "answers": [["down", 1.0], ["at table", 0.3], ["skateboard", 0.3], ["table", 0.3]]}

# question{
# "question_id" : int,
# "image_id" : int,
# "question" : [word],
# "answers" : [tuple(answer,score)] 按照score降序
# "correct" : str
# }

def main():
    # load data
    #list of dict for train, this time only from data/vqa02/raw-train2014.json
    fname='../data/vqa02/train2014-raw.json'
    print('[Load] {}'.format(fname))
    train_data=json.load(open(fname))

    # list of dict for evaluate, data/raw-val2014.json
    fname = '../data/vqa02/val2014-raw.json'
    print('[Load] {}'.format(fname))
    val_data=json.load(open(fname))

    # determine answer candidates
    ans_freq = Counter()
    for pair in train_data:
        ans_freq.update(dict(pair['answers']))#将“answers”属性转为dict：answer:score, update可以累加answer的score, 2维list可转为dict
    #__C.MIN_ANS_FREQ = 16, 按答案分总分序排序的list， 并且剔除出现总分小于cfg.MIN_ANS_FREQ的answer(小于cfg.MIN_ANS_FREQ无法成为候选答案)
    #返回一个TopN列表[(key, value)] 如果n没有被指定，则返回所有元素, index to answer, 且按answer的总分降序排序，且总分>cfg.MIN_ANS_FREQ
    itoa = [a for a, c in ans_freq.most_common() if c > cfg.MIN_ANS_FREQ]#index to answer
    #dict for answer:index(index越小说明answer总分越大，且总分>cfg.MIN_ANS_FREQ)
    atoi = {a:i for i, a in enumerate(itoa)}
    print('[Info] answer candidates count: {}'.format(len(itoa)))
    if cfg.DEBUG:
        print('[Debug] top answer')
        print(' '.join(itoa[:10]))# 10个分数最高的answer


    # determine vocabulary for question
    word_freq = Counter()
    for pair in train_data:#list of dict for train
        word_freq.update(pair['question'])
    word_freq = word_freq.most_common()#[(word,appearance),...]
    #__C.MIN_WORD_FREQ = 1,只有出现>=2次，word才能加入字典，内部按word出现次数降序排序, index to word
    itow = [w for w, c in word_freq if c > cfg.MIN_WORD_FREQ]
    print('[Info] Reserved words count: {}/{}({:%})'.format(len(itow), len(word_freq), len(itow)/len(word_freq)))
    assert('<UNK>' not in itow)#判断为FALSE时，奔溃，所以要求判断为真
    assert('<PAD>' not in itow)
    itow = ['<PAD>'] + itow + ['<UNK>']#加上开头与结尾，index to word
    wtoi = {w: i for i, w in enumerate(itow)}#word to index
    print('[Info] vocabulary size: {}'.format(len(itow)))
    if cfg.DEBUG:
        print('[Debug] top word')
        print(' '.join(itow[:10]))
        print('[Debug] last word')
        print(' '.join(itow[-10:]))

    # encode question
    # question index matrix and question_id matrix
    # 将question word matrix转为index matrix，每一行为一个问题，每一行中有14个数，question word的index向右靠齐，左边填0
    train_que, train_que_id = encode_que(train_data, wtoi, itow) #question index 2d-ndarray and question_id 1d-ndarray

    #返回一个2d-ndarray，内有len(data)个1d-ndarray，每个内部1d-ndarray中为 所有候选answer(3196)的ground-truth score
    #返回一个1d-ndarray，分别是每个问题的correct answer在候选答案中对应的index，若没有在候选答案中则为最大的index+1
    train_ans, train_correct = encode_ans(train_data, atoi, 'train2014')
    train_img_id=encode_image(train_data)

    # encode question
    # question index matrix and question_id matrix
    # 将question word matrix转为index matrix，每一行为一个问题，每一行中有14个数，question word的index向右靠齐，左边填0
    val_que, val_que_id = encode_que(val_data, wtoi, itow)#question index 2d-ndarray and question_id 1d-ndarray

    #返回一个2d-ndarray，内有len(data)个1d-ndarray，每个内部1d-ndarray中为 所有候选answer(3196)的ground-truth score
    val_ans, val_correct = encode_ans(val_data, atoi, 'val2014')
    val_img_id=encode_image(val_data)

    # Save
    codebook = {}
    codebook['itoa'] = itoa
    codebook['itow'] = itow
    codebook['atoi'] = atoi
    codebook['wtoi'] = wtoi

    codebook_fname = '../data/vqa02/codebook.json'
    print('[Store] {}'.format(codebook_fname))
    json.dump(codebook, open(codebook_fname, 'w'))

    train_h5 = '../data/vqa02/train2014-paired.h5'
    print('[Store] {}'.format(train_h5))
    with h5py.File(train_h5, 'w') as f:
        group = f.create_group('train')
        group.create_dataset('que_id', dtype='int64', data=train_que_id)
        group.create_dataset('que', dtype='int64', data=train_que)

        group.create_dataset('ans', dtype='float16', data=train_ans)
        group.create_dataset('correct', dtype='int64', data=train_correct)
        group.create_dataset('img_id', dtype='int64', data=train_img_id)
        # train_que中的每一行的question
        # 对应 train_que_id中同一行的question_id
        # 对应 train_ans中同一行的的答案分布
        # 对应 train_correct中的correct answer index
        # 对应 train_image_id中同一行的image_id

    val_h5 = '../data/vqa02/val2014-paired.h5'
    print('[Store] {}'.format(val_h5))
    with h5py.File(val_h5, 'w') as f:
        group = f.create_group('val')
        group.create_dataset('que_id', dtype='int64', data=val_que_id)
        group.create_dataset('que', dtype='int64', data=val_que)

        group.create_dataset('ans', dtype='float16', data=val_ans)
        group.create_dataset('correct', dtype='int64', data=val_correct)
        group.create_dataset('img_id', dtype='int64', data=val_img_id)



def encode_que(data, wtoi, itow):#data：train_data
    N = len(data)
    question_id = np.zeros((N,), dtype='int64')#question_id matrix(N,)
    que = np.zeros((N, cfg.MAX_QUESTION_LEN), dtype='int64')#question matrix(N, cfg.MAX_QUESTION_LEN)

    unk_cnt = 0#长度过长的问题的个数
    trun_cnt = 0#没有在dict内的word数
    unk_idx = wtoi.get('<UNK>')#在word字典中找不到该word的时候的word index
    for i,sample in enumerate(data):
        question_id[i] = sample['question_id']

        words = sample['question']#'question'属性为list of word
        nword = len(words)
        if nword > cfg.MAX_QUESTION_LEN:#截取长度过长的问题
            trun_cnt += 1
            nword = cfg.MAX_QUESTION_LEN
            words = words[:nword]
        for j, w in enumerate(words):
            #将question word matrix转为index matrix，每一行为一个问题，每一行中有14个数，question word的index向右靠齐，左边填0
            word_index=wtoi.get(w, unk_idx)
            que[i][cfg.MAX_QUESTION_LEN-nword+j] = word_index
            unk_cnt += (1 if word_index==unk_idx else 0)

    print('[Info] Truncated question count: {}'.format(trun_cnt))
    print('[Info] Unknown word count: {}'.format(unk_cnt))

    print('[Debug] question index')
    samples = random.sample(que.tolist(), k=5)
    for s in samples:
        print(' '.join([itow[index] for index in s]))

    print('[Debug] question id')
    samples = random.sample(question_id.tolist(), k=5)
    for s in samples:
        print(s)

    return que, question_id #question index 2d-ndarray and question_id matrix 1d-ndarray


# 返回一个2d-ndarray，内有len(data)个1d-ndarray，每个内部1d-ndarray中为 所有候选answer的ground-truth score
# 每一行内，该train_data的这个dict对应的这一行中，这个dict中出现的答案的位置内容为该答案的分数，其他为0
#返回一个1d-ndarray，分别是每个问题的correct answer在候选答案中对应的index，若没有在候选答案中则为最大的index+1
def encode_ans(data, atoi, split):#data：train_data
    #此时每个question都有多个答案，需要拟合答案的分布
    ans = np.zeros((len(data), len(atoi)+1), dtype='float32')
    correct_index=np.zeros((len(data),), dtype='int64')
    # answers: [["net", 1.0], ["netting", 0.3], ["mesh", 0.3]]
    for i, answers in enumerate(map(itemgetter('answers'), data)):
        for answer, score in answers:
            ans[i][atoi.get(answer,len(atoi))] += score
    N=len(atoi)
    num_not_in_ans=0
    for i, correct in enumerate(map(itemgetter('correct'), data)):
        correct_index[i]=atoi.get(correct,N)
        if correct_index[i]==N:
            num_not_in_ans+=1
    print('{}/{} ({}%) correct answers in {} is not in candidate answers'.format(num_not_in_ans,len(data),100.0*num_not_in_ans/len(data),split))
    correct_index = np.argmax(ans, axis=1)  # ndarray (bs,)

    print('[Debug] answer distribution')
    print("[ans size] ",ans.shape)
    # samples = random.sample(ans.tolist(), k=5)
    # print("[ans distribution] 5 ans sum: ",np.sum(samples,axis=1,dtype=np.float32))
    # for s in samples:
    #     print(s)

    print('[Debug] correct index')
    samples = random.sample(correct_index.tolist(), k=5)
    for s in samples:
        print(s)

    # indexs = np.argmax(ans, axis=1)  # ndarray (bs,)
    # right = list(map(lambda x, y: 1 if x == y else 0, indexs, correct_index)).count(1)
    # print("{}/{} {:.2f} is right in candidate".format(right,correct_index.shape[0],100.0*right/correct_index.shape[0]))
    return ans, correct_index

#返回image_id list
def encode_image(data):
    N = len(data)
    image_id=np.zeros((N,),dtype='int64')#image_id matrix(N,)
    for i, img_id in enumerate(map(itemgetter('image_id'),data)):
        image_id[i]=img_id
    return image_id


if __name__ == '__main__':
    main()