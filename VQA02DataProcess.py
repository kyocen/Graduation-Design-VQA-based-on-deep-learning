import os
import re
import json
import random
import argparse
from collections import Counter
from operator import itemgetter

import nltk
import progressbar
from config import cfg

# question{
# "question_id" : int,
# "image_id" : int,
# "question" : [word],
# "answers" : [tuple(answer,score)] 按照score降序
# "ans_num" : [tuple(answer,appearance)] 按照appearance降序
# "correct" : str
# }
def main():#加入了“answers”属性的list of question dict，answers属性为所有答案的降序list of tuple(answer,score of appearance)
    for split in ("train2014", "val2014"):#将原始数据处理成 data/raw-{}.json
        pairs=load_vqa_data(split)
        fname='../data/vqa02/{}-raw.json'.format(split)
        print("[Store] {}".format(fname))
        json.dump(pairs,open(fname,'w'))


#返回加入了“answers”属性的list of question dict，answers属性为所有答案的降序list of tuple(answer,score of appearance)
def load_vqa_data(split):#split_name为文件名, VQA-COCO数据集中的打包
    qfname='../data/vqa02/v2_OpenEnded_mscoco_{}_questions.json'.format(split)
    afname='../data/vqa02/v2_mscoco_{}_annotations.json'.format(split)

    print('[Load] load quesiton from "{}"'.format(qfname))
    vqa_pair=json.load(open(qfname,'r'))['questions']#list of question dict

#question{
#"question_id" : int,
#"image_id" : int,
#"question" : str
#}
    print('[Info] tokenizing')
    bar=progressbar.ProgressBar()
    for pair in bar(vqa_pair):#pair is the dict of question illustrated above
        qtext=pair['question'].lower().strip()
        if(qtext[-1])=='?':
            qtext=qtext[:-1]
        pair['question']=nltk.word_tokenize(qtext)

    if cfg.DEBUG:
        print('[Debug] question after tokenizing')
        # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，这里为获取question属性的iterator
        questions=map(itemgetter('question'), vqa_pair)#iterator of list of question
        sample_ques = random.sample(list(questions), k=5)  # 从指定序列中随机获取k个不重复的样本
        # 字符串(以空格为分界)、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
        # 'sep'.join(iteration) sep：分隔符,可以为空 iteration：要连接的元素序列、字符串、元组、字典
        print('\n'.join([' '.join(q) for q in sample_ques]))

    print('[Load] load annotation data from "{}"'.format(afname))
    anns = json.load(open(afname))['annotations']#list of annotations dict
    # annotation{
    # "question_id" : int,
    # "image_id" : int,
    # "question_type" : str,
    # "answer_type" : str,
    # "answers" : [answer], 10个answer 其中可能会有一样的 所以len(set(answer)) 不一定
    # "multiple_choice_answer" : str
    # }
    # answer{
    # "answer_id" : int,
    # "answer" : str,
    # "answer_confidence": str
    # }

    qid_anns={a['question_id']: a for a in anns}#dict: question_id->annotations dict
    bar = progressbar.ProgressBar()
    for q in bar(vqa_pair):#vqa_pairs is the list of question dict, q is the dict of question
        answers=qid_anns.get(q['question_id']).get('answers')#该问题q对应的答案选项：list of answer dict
        for item in answers:
            item['answer']= norm_answer(item['answer'])#化简answer

        ans_text = set(map(itemgetter('answer'), answers))  # 该question对应的所有答案的set:set of answer

        ans_score = []#[(answer, score)]
        for at in ans_text:  # at为该question对应的所有答案之一
            accs = []
            for gt in answers:  # gt为该question对应的answer idct之一 answer该问题q对应的答案选项：list of answer dict
                other_gt = [a for a in answers if a != gt]  # 除gt之外的该question对应的answer idct
                matched_gt = [a for a in other_gt if a['answer'] == at]  # 除gt之外的且answer为at的answer dict
                accs.append(min(1.0, len(matched_gt) / 3))
            # 这些答案都是人标注的，出现次数越高，被认可越高，准确率越好，出现3次就算一定对，at和这个答案出现的次数（次数越大，sum(accs)/len(accs)越大）
            ans_score.append((at, sum(accs) / len(accs)))
        ans_score = sorted(ans_score, key=itemgetter(1), reverse=True)  # 根据准确率排序

        ans_freq=Counter()
        ans_freq.update(list(map(itemgetter('answer'), answers)))
        q['ans_num']=ans_freq.most_common()
        q['answers'] = ans_score
        q['correct']=norm_answer(qid_anns.get(q['question_id']).get('multiple_choice_answer'))

    if cfg.DEBUG:
        print('[Debug] answers with score')
        ans=map(itemgetter('answers'), vqa_pair)#iterator of list of answer
        sample_ans = random.sample(list(ans), k=5)  # 从指定序列中随机获取k个不重复的样本
        # 字符串(以空格为分界)、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
        # 'sep'.join(iteration) sep：分隔符,可以为空 iteration：要连接的元素序列、字符串、元组、字典
        for q in sample_ans:
            print(q)
        print('[Debug] five vqa pair')
        samples=random.sample(vqa_pair,k=5)
        for s in samples:
            print(s)

    return vqa_pair  # 加入了“answers”属性的list of question dict，answers属性为所有答案的降序list of tuple(answer,score)

# question{
# "question_id" : int,
# "image_id" : int,
# "question" : [word],
# "answers" : [tuple(answer,score)] 按照score降序
# "ans_num" : [tuple(answer,appearance)] 按照appearance降序
# "correct" : str
# }

def norm_answer(ans):#化简answer string
    ans = ans.replace('\n', ' ').replace('\t', ' ').strip()#去掉\n,\t，去掉前后空格
    ans = process_punct(ans)#去掉标点
    ans = process_digit_article(ans)#去掉冠词，扩展缩写，处理数字
    return ans


# borrow from vqaEval.py
m_contractions = {#扩展缩写
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"}
m_manual_map = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                'eight': '8', 'nine': '9', 'ten': '10'}#数字
m_articles = ['a', 'an', 'the']#冠词
m_period_strip = re.compile("(?!<=\d)(\.)(?!\d)")#匹配句号，来去掉句号，(?!...)表示这一部分不能是...
m_comma_strip = re.compile("(\d)(\,)(\d)")#匹配数字之间的，，来去掉，，比如123,456
m_punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+',
           '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']#标点


def process_punct(in_text):#去掉标点
    out_text = in_text
    for p in m_punct:
        if (p + ' ' in in_text or ' ' + p in in_text or re.search(m_comma_strip, in_text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = m_period_strip.sub("", out_text, re.UNICODE)#去掉句号
    return out_text


def process_digit_article(in_text):#去掉冠词，扩展缩写，处理数字
    out_text = []
    for word in in_text.lower().split():
        if word not in m_articles:
            word = m_manual_map.get(word, word)
            word = m_contractions.get(word, word)
            out_text.append(word)
    return ' '.join(out_text)


if __name__ == '__main__':
    main()


