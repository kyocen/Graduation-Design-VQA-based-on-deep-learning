# config system imitating Faster RCNN

import os
import os.path as osp
from easydict import EasyDict as edict
import numpy as np

#root dict is /home/pzy/project/vqa-concept/
__C = edict()
cfg = __C

# print debug information
__C.DEBUG = True

# Path to data
__C.DATA_DIR = 'data'

# Path to vqa tools
__C.VQA_DIR = 'vqa-tools'

# Path to log files
__C.LOG_DIR = 'log'

# Path to Visual Gnome data
__C.VG_DIR = 'data/vg'

# Splits of VQA to use during training
__C.TRAIN = edict()
__C.TRAIN.SPLITS = ('train2014', 'vg')

# Splits of VQA to use during testing
__C.TEST = edict()
__C.TEST.SPLITS = ('val2014',)

# Minimun frequency of the answer which can be choosed as a candidate
__C.MIN_ANS_FREQ = 16

# Minimun frequency of the word which can be included in the vocaburay
__C.MIN_WORD_FREQ = 1

# Maximun length of the question which the model take as input
# the question longer than that will be truncated
__C.MAX_QUESTION_LEN = 14

# Random seed
__C.USE_RANDOM_SEED = True
__C.SEED = 42

# source name of feature ('bottomup' or 'densecap')
__C.FEATURE_SOURCE = 'bottomup'

# number of boxes per image
__C.NUM_BOXES = 36

# name of pretrained embedding
__C.WORD_EMBEDDINGS = 'glove.6B.300d.txt'

# Use soft sigmoid loss
__C.SOFT_LOSS = True

# load all data into memory, which will be faster when
# iterating dataset many times
__C.LOAD_ALL_DATA = True


def get_feature_path(split, fea_name):#根据特征名和split得到该图片特征的path
    if split == 'test-dev2015':
        split = 'test2015'
    return '{}/image-feature/{}/{}_{}_{}.npy'.format(__C.DATA_DIR, __C.FEATURE_SOURCE, split, __C.NUM_BOXES, fea_name)


def get_emb_size():
    emb_size = 300
    if __C.WORD_EMBEDDINGS:
        emb_names = __C.WORD_EMBEDDINGS.split('+')
        emb_size = 0
        for emb_name in emb_names:
            emb_path = '{}/word-embedding/{}'.format(__C.DATA_DIR, emb_name)
            with open(emb_path) as f:
                line = f.readline()
            emb_size += len(line.split()) - 1
    return emb_size

##############################################################################
# Copy from RCNN


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

