VQA01DataProcess是对数据集的基础的处理，将数据配对
VQA01ImageProcess 提取图片特征，获得embedding dict
VQA01dataset 将数据包装成dataset
VQA01Baseline 用来训练模型并且进行评估

毕设：
VQA02DataProcess是对数据集的基础的处理，将数据配对包括question_id, question, [answer], image_id, correct_answer
VQA02getdata是将question处理成2d index matrix, 并选出候选答案和提取word dict，将[answer]处理成answer index->score
VQA02ImageProcess 提取图片特征到文件
VQA02dataset 将数据包装成dataset
CSFMODEL 一整个模型
modules 要用到的模块，包括MFH，CSF
resnet resnet源码，和我修改之后的resnet模型
VQA02train 用来训练模型，得到结果

对照试验：
num of CSF layers -l
use freq in answer rather than grade -f
model -m c: CSFMODEL m: MFHMODEL b: MFHBaseline
sub model -s cs/csf
require_gard -g 1/0

parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=10)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=7e-4)
parser.add_argument("-wd", type=float, action="store", help="weight decay", default=0)
parser.add_argument("-epoch", type=int, action="store", help="epoch", default=10)
parser.add_argument("-e", type=float, action="store", help="extend for score", default=1.0)
parser.add_argument('--print-freq', '-p', default=2000, type=int, metavar='N', help='print frequency (default: 1000)')

parser.add_argument("-gpu", type=int, action="store", help="gpu_index", default=1)
parser.add_argument("-f",type=int, action="store",help="use freq in answer rather than grade",default=0)
parser.add_argument("-gru",type=int,action="store",help="use GRU rather than LSTM",default=0)#gru or lstm
parser.add_argument("-co",type=int, action="store",help="co-attention",default=0)
parser.add_argument("-l", type=int, action="store", help="num of CSF layers", default=0)
parser.add_argument("-m",type=str,nargs=1,choices=['c','m','b'],help="model",default='b')#c: CSFMODEL m: MFHMODEL b: MFHBaseline
parser.add_argument("-s",type=str,nargs=1,choices=['cs','csf'],help="model",default='csf')#cs: CS csf: CSF
parser.add_argument("-g",type=int, action="store",help="fine tune on the conv",default=0)
parser.add_argument("-sig",type=int,action="store",help="use sigmoid rather than softmax",default=0)#cs: CS csf: CSF
parser.add_argument("-dey", type=float, action="store", help="learning rate decay", default=1)
parser.add_argument("-loss", type=int, action="store", help="adjust learning rate by loss", default=0)

baseline 和 CSFMODEL的差别在于baseline可以选择用co-attention，并且用LSTM，CSFMODEL用GRU
baseline 和 MFHMODEL的差别在于MFHMODEL在得到image feature tensor之后并没有直接转为image feature vector而是再与question feature做了一次权重计算然后加权和

baseline freq=0 layer=0 cs acc:53.89

baseline freq=0 layer=0 co_att acc:48.20
baseline freq=0 layer=0 cs grad=1 acc:51.46

baseline freq=0 layer=1 csf acc:51.65
baseline freq=0 layer=2 csf acc:
baseline freq=0 layer=3 csf acc:

baseline freq=0 layer=1 cs  acc:53.72
baseline freq=0 layer=2 cs  acc:55.50
baseline freq=0 layer=3 cs  acc:52.29

baseline freq=0 layer=2 cs  grad=1 acc:55.64
baseline freq=0 layer=0 cs sig gru acc:54.15

baseline freq=0 layer=0 cs g=0 dey=1 acc:53.71
baseline freq=0 layer=0 cs g=0 dey=0.5 acc:53.89
baseline freq=0 layer=0 cs g=0 dey=0.1 acc:53.75

baseline freq=0 layer=0 cs g=0 dey=1 sig=1 acc:53.98


MFHMODEL freq=0 layer=0 cs  acc:55.18

MFHMODEL freq=0 layer=0 cs co_att acc:48.2
MFHMODEL freq=0 layer=0 cs grad=1 acc:55.50

MFHMODEL freq=0 layer=1 csf acc:
MFHMODEL freq=0 layer=2 csf acc:
MFHMODEL freq=0 layer=3 csf acc:

MFHMODEL freq=0 layer=1 cs  acc:55.44
MFHMODEL freq=0 layer=2 cs  acc:55.48
MFHMODEL freq=0 layer=3 cs  acc:55.41

baseline freq=0 layer=2 cs  grad=1 dey=0.5 sig=1 acc:54.54
MFHMODEL freq=0 layer=2 cs  grad=1 dey=0.5 sig=1 acc:58.34
