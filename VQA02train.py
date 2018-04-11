# 首先将image和question结合，再将question与answer结合，image和question结合主要通过推理来提取所需特征，
# question与answer结合用于从有限但是大量的候选答案中选出有可能的几个，即answer与question对应的几个答案
# 再通过image和question结合得到的fuse来选出最终的答案，所以用answer score分布和KL散度来学习是合理的，
# 每个question的ground-truth应该是少量几个答案的score为0-1之间，其他答案的分数都为0
import argparse
import sys
import os
import shutil
import time
import logging
import datetime
import json
import math
from importlib import import_module

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import progressbar
import numpy as np
from torch.autograd import Variable
from visdom import Visdom

from VQA02dataset import VQA02Dataset
from config import cfg, cfg_from_file, cfg_from_list
from CSFMODEL import CSFMODEL

parser = argparse.ArgumentParser(description="VQA")
parser.add_argument("-gpu", type=int, action="store", help="gpu_index", default=1)
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=10)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=7e-4)
parser.add_argument("-wd", type=float, action="store", help="weight decay", default=0)
parser.add_argument("-epoch", type=int, action="store", help="epoch", default=50)
parser.add_argument("-l", type=int, action="store", help="num of CSF layers", default=3)
parser.add_argument("-e", type=float, action="store", help="extend for score", default=5.0)
parser.add_argument('--print-freq', '-p', default=2000, type=int, metavar='N', help='print frequency (default: 1000)')

args = parser.parse_args()

if cfg.USE_RANDOM_SEED:
    torch.manual_seed(cfg.SEED)  # Sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        cfg.SEED)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available

BATCH_SIZE = args.bs

logger = logging.getLogger('vqa')  # logging name
logger.setLevel(logging.DEBUG)  # 接收DEBUG即以上的log info


def main():
    fh = logging.FileHandler('./current.log')  # log info 输入到文件
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)  # log info 输入到屏幕
    sh.setLevel(logging.DEBUG)

    fmt = '[%(asctime)-15s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)  # 设置每条info开头格式
    logger.addHandler(fh)  # 把FileHandler/StreamHandler加入logger
    logger.addHandler(sh)

    # select device
    torch.cuda.set_device(args.gpu)
    logger.debug('[Info] use gpu: {}'.format(torch.cuda.current_device()))

    # data
    logger.debug('[Info] init dataset')

    train_set = VQA02Dataset('train2014',args.e)
    # Data loader Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )  # If True, the data loader will copy tensors into CUDA pinned memory before returning them

    val_set = VQA02Dataset('val2014',args.e)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # initialize word embedding with pretrained
    word_vec, emb_size = extract_embedding('../data/glove/glove.6B.300d.txt')  # dict word->embedding
    logger.debug('[Info] embedding size: {}'.format(emb_size))

    # 建立模型
    model = CSFMODEL(args.l, len(train_set.codebook['itow']), len(train_set.codebook['itoa'])+1, emb_size)

    total_param = 0
    for param in model.parameters():  # Returns an iterator over module parameters
        total_param += param.nelement()  # Returns the total number of elements in the input tensor.参数中所有元素的个数
    logger.debug('[Info] total parameters: {}M'.format(math.ceil(total_param / 2 ** 20)))
    # Return the the smallest integer value greater than or equal to x.

    # self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
    # nn.Embedding weight: the learnable weights of the module of shape (num_words, embedding_dim)
    # model.we.weight(tensor(num_words, embedding_dim))就是一张没有初始化的embedding lookup table, 由于没有初始化，所以这里就是zero matrix
    emb_table = model.we.weight.data.numpy()  # 2d ndarray word index->embedding vector
    assert '<PAD>' not in word_vec
    vaild_emb = 0  # 有效embedding的个数
    for i, word in enumerate(train_set.codebook['itow']):
        if word in word_vec:
            emb_table[i] = word_vec[word]
            vaild_emb += 1
    logger.debug('[debug] word embedding filling count: {}/{}'.format(vaild_emb, len(
        train_set.codebook['itow'])))  # vaild embedding/num of question word
    # 初始化embedding look up table, 由于question是以index of word形式传入的，所以embedding look up table 只要保存index到embedding的对应关系即可
    model.we.weight = nn.Parameter(torch.from_numpy(emb_table))

    # BCEWithLogitsLoss：This loss combines a Sigmoid layer and the BCELoss in one single class.
    # 输入为两个3196的vector
    # 得到的答案为3196维，标准答案为vector，3196中每个都有一个score，相当于对每一个候选答案都做一个BCE，然后对所有维度做平均，再对整个batch做平均
    if torch.cuda.is_available():
        model.cuda()
        criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    # It enables benchmark mode in cudnn.
    # benchmark mode is good whenever your input sizes for your network do not vary.
    # This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
    # But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
    cudnn.benchmark = True

    # train
    logger.debug('[Info] start training...')
    is_best = False
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(1, args.epoch + 1):  # 每一个epoch遍历所有batch
        loss = train(train_loader, model, criterion, optimizer, epoch, val_loader)
        acc = validate(val_loader, model, criterion, epoch)  # 所有batch，所有样本的总和accuracy
        if acc > best_acc:
            is_best = True
            best_acc = acc
            best_epoch = epoch

        logger.debug('[epoch {}]: loss : {}'.format(epoch, loss))
        logger.debug('Evaluate Result:\t' 'Acc  {0}\t' 'Best {1} ({2})'.format(acc, best_acc, best_epoch))

        if is_best:
            state = {
                'epoch': epoch,
                # Returns a dictionary containing a whole state of the module.
                # Both parameters and persistent buffers (e.g. running averages) are included.
                # Keys are corresponding parameter and buffer names.
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            best_path = './model-best.pkl'
            torch.save(state, best_path)
    logger.debug('Evaluate Result:\t' 'best accuracy:  {0}\t' 'best epoch{1}'.format(best_acc, best_epoch))


def extract_embedding(filepath):
    logger.debug('[Load] ' + filepath)
    with open(filepath, 'r') as f:
        word_vec_txt = [l.rstrip().split(' ', 1) for l in
                        f.readlines()]  # rstrip()去掉末尾的空白和换行符 2d list [[word,embedding]...]
    vocab, vec_txt = zip(*word_vec_txt)  # tuple of words, tuple of embedding (word,...) (embedding string,...)
    embedding_size = len(vec_txt[0].split())
    # fromstring faster than loadtxt
    vector = np.fromstring(' '.join(vec_txt), dtype='float32',
                           sep=' ')  # 将string转为float32，fromstring用来分割文本并转化类型 1d ndarray of float32
    vector = vector.reshape(-1, embedding_size)  # ndarray of 2d (num of word, embedding length)
    word_vec = {}  # dict word->embedding
    for i, word in enumerate(vocab):
        word_vec[word] = vector[i]
    return word_vec, embedding_size


def train(train_loader, model, criterion, optimizer, epoch,val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()  # Sets the module in training mode
    end = time.time()  # 返回当前时间，以秒为单位
    # [question [index of word] ndarray 1d, image feature  3d ndarray (2048,7,7),
    # 1d ndarray [score(float32) of N candidate answers for this question], #int64  correct answer index]
    # 一个sample是一个经过处理的iterator of batch(相当于iterator of the list of iterator)，2d tensor，sample[i]是所有单个样本第i个的属性的集合的iterator：1d tensor，！！把他当做单个样本处理即可！！
    for i, sample in enumerate(train_loader,1):  # sample即一个batch，dataloader is a iterator of the list of iterator of batch
        data_time.update(time.time() - end)
        # [question_id,ndarray(image feature),question:[list of word index],ndarray(object feature),answers(两种模式)]
        sample_var = [Variable(d).cuda() for d in
                      list(sample)[0:3]]  # Variable list of iterator [iterator for img, que, [obj], ans]

        # input： # img: [bs,2048,7,7] que: (bs,14)
        # output：3092的1d vector
        # [question [index of word] ndarray 1d, image feature  3d ndarray (2048,7,7),
        # 1d ndarray [score(float32) of N candidate answers for this question], #int64  correct answer index]
        score = model(*sample_var[:-1])  #que: (bs,14) img: [bs,2048,7,7]
        loss = criterion(score, sample_var[-1])  # 虽然是处理一个batch，但loss是一个scalar，是batch内所有样本的loss的均值，但是是一个tensor

        losses.update(loss.data[0])  # loss.data和sample[0]都是tensor sample[0].size()会返回一个object，sample[0].size(0)会返回一个值

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.debug(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

            validate(train_loader, model, criterion, epoch)
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    model.eval()  # 调整到eval模式

    # sample: (question, img_feature, ans, correct)
    amount = 0
    right = 0
    bar=progressbar.ProgressBar()
    for i, sample in enumerate(bar(val_loader), 1):
        sample_var = [Variable(d).cuda() for d in list(sample)[0:3]]  # Variable list of iterator [iterator for img, que, [obj], ans]
        amount += sample[0].size(0)
        # input： # img: [bs,2048,7,7] que: (bs,14)
        # output：3097的1d vector(bs,3097)
        score = model(*sample_var[:-1])  # img: [bs,2048,7,7] que: (bs,14) #score is a Variable of the list of scores
        _, indexs = torch.max(score.data, dim=1)#tensor (bs,)
        correct_batch = list(map(lambda x, y: 1 if x == y else 0, indexs, sample[3])).count(1)
        right += correct_batch
        if i in [20,40,60,80,100]:
            print("[size test] sample[0] size: {}\n score size {}\n indexs: {}\n sample[3] size: {}\n".format(sample[0].size(),score.size(),indexs,sample[3]))
        if i==100:
            break
    accuracy=100.0*float(right)/float(amount)
    print('[%5d] accuracy: %.3f' % (epoch, 100.0*float(right)/float(amount)))
    model.train()
    return accuracy


def save_checkpoint(state, is_best, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        'model_best.pkl')  # Copy the contents of the file named src to a file named dst, 如果dst已经存在，它会被替换


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 每次的值
        self.avg = 0  # 从头开始到现在值的平均
        self.sum = 0  # 从头开的值的总和
        self.count = 0  # 从头开始值的个数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
