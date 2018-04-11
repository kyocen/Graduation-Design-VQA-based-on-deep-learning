# Download the VQA Questions from http://www.visualqa.org/download.html
import json
import os
import re
import glob
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# read feature file 从图片的npy文件中提取dictionary，以文件名为key，以feature为value, 所有batch和在一起
def load_feature(path):
    dict = {}
    pattern = os.path.join(path, '*.npy')
    for i, filepath in enumerate(glob.glob(pattern), 1):
        feature_batch = np.load(filepath)
        for key in feature_batch.item():#feature_batch.item() is a dict
            dict[key] = feature_batch.item()[key]
    print(len(dict))

    return dict

# [{'ques_id': question_id(int), 'img_path': image_path(str), 'question': question(str), 'MC_ans': mc_ans(list of 18 str), 'ans': ans(str)}]
# [[sample(filename(str), question(str), answer(str), split(str), label(bool))]] 248349*18 return list of list of dict
def construct_sample(sample_set,split):
    samples=[]
    for x in sample_set:
        onesample = []
        for i in range(18):
            sample = {}
            sample['filename'] = x['img_path']
            sample['question'] = x['question']
            sample['answer'] = x['MC_ans'][i]
            sample['split'] = split
            sample['label'] = int(sample['answer']==x['ans'])
            onesample.append(sample)
        samples.append(onesample)
    print("samples: ",np.asarray(samples).shape)
    return samples

## get dictionary 从所有question和answer中构造字典
def get_dictionary(dataset):
    dictionary = set()
    for samples in dataset:
        for sample in samples:
            question = sample['question']
            add_to_dict(question, dictionary)
            answer = sample['answer']
            add_to_dict(answer, dictionary)

    return dictionary

## add to dictionary 通过正则表达式从question和answer中提取单词加入字典
def add_to_dict(str, dictionary):
    str = str.lower()
    txt = re.findall(r"[:]|[^\w\s]|\w+", str)

    for word in txt:
        if word not in dictionary:
            dictionary.add(word)


## parse GloVe to dict, glove中每个word后面跟300个数字最为一个line，将所有embedding整理成字典
def parse_glove(path):
    w2v = {}
    with open(path, encoding="utf-8",mode='r') as f:
        for line in f:
            words = line.split(' ')
            word = words[0]
            numbers = words[1:]
            w2v[word] = np.asarray(numbers).astype(float)

    return w2v

# {
# "info" : info,
# "data_type": str,
# "data_subtype": str,
# "annotations" : [annotation],
# "license" : license
# }
# annotation{
# "question_id" : int,
# "image_id" : int,
# "question_type" : str,
# "answer_type" : str,
# "answers" : [answer],
# "multiple_choice_answer" : str
# }
# answer{
# "answer_id" : int,
# "answer" : str,
# "answer_confidence": str
# }
# question{
# "question_id" : int,
# "image_id" : int,
# "question" : str
# }
#E:\workPlaceForPython\dataset\COCO\train2014/COCO_train2014_000000115654.jpg
def construct_dataset():
    train = []
    eval = []
    imdir = '../dataset/COCO/%s/COCO_%s_%012d.jpg'

    print('Loading annotations and questions...')
    train_anno = json.load(open('../dataset/VQA/v1 Real Image/Annotations_Train_mscoco/mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open('../dataset/VQA/v1 Real Image/Annotations_Val_mscoco/mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open('../dataset/VQA/v1 Real Image/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open('../dataset/VQA/v1 Real Image/Questions_Val_mscoco/MultipleChoice_mscoco_val2014_questions.json', 'r'))

    subtype = 'train2014'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']#正确答案
        question_id = train_anno['annotations'][i]['question_id']#question_id int
        image_path = imdir % (subtype, subtype, train_anno['annotations'][i]['image_id'])#int

        question = train_ques['questions'][i]['question']#string
        mc_ans = train_ques['questions'][i]['multiple_choices']#list of 18 string, includes the correct answer

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        image_path = imdir % (subtype, subtype, val_anno['annotations'][i]['image_id'])

        question = val_ques['questions'][i]['question']
        mc_ans = val_ques['questions'][i]['multiple_choices']

        eval.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    print('Training sample %d, Evaluating sample %d...' % (len(train), len(eval)))

    json.dump(train, open('../dataset/VQA/v1 Real Image/vqa_raw_train.json', 'w'))
    json.dump(eval, open('../dataset/VQA/v1 Real Image/vqa_raw_eval.json', 'w'))


## resize images - done once and for ever
def resize_images(im_size, src, dst):
    pattern = os.path.join(src, '*.jpg')
    #glob.glob()返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径
    for filepath in glob.glob(pattern):
        print(os.path.basename(filepath))
        im = Image.open(filepath)
        im = im.resize((im_size, im_size), Image.ANTIALIAS)
        im.save(os.path.join(dst, os.path.basename(filepath)))

IM_SIZE=224


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # # input json
    # parser.add_argument('--download', default='False', help='Download and extract data from VQA server')
    # parser.add_argument('--split', default=1, type=int,
    #                     help='1: train on Train and eval on Val, 2: train on Train+Val and eval on evaluate')
    #
    # args = parser.parse_args()
    # params = vars(args)
    # print
    # 'parsed input parameters:'
    # print
    # json.dumps(params, indent=2)
    #
    ## resize images
    # resize_images(IM_SIZE, '../dataset/COCO/train2014/train2014/', '../dataset/COCO/train2014_224/')
    # resize_images(IM_SIZE, '../dataset/COCO/val2014/val2014/', '../dataset/COCO/val2014_224/')
    construct_dataset()
