import sys
from config import cfg
sys.path.insert(0, './vqa-tools/PythonHelperTools/vqaTools')
sys.path.insert(0, './vqa-tools/PythonEvaluationTools')
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

import json
import os
import tempfile
from operator import itemgetter


class VQATool(object):
    _instances = {}

    def __init__(self):
        self.que_fptn = '{}/v2_OpenEnded_mscoco_{}_questions.json'
        self.ann_fptn = '{}/v2_mscoco_{}_annotations.json'


    def get_que_path(self, vqa_dir, split):# 'vqa-tools' 'train2014'/'val2014'
        return self.que_fptn.format(vqa_dir, split)


    def get_ann_path(self, vqa_dir, split):
        return self.ann_fptn.format(vqa_dir, split)


    def get_vqa(self, vqa_dir, split):
        if split not in self._instances:
            que_fname = self.get_que_path(vqa_dir, split)
            ann_fname = self.get_ann_path(vqa_dir, split)
            self._instances[split] = VQA(ann_fname, que_fname)
        return self._instances[split]

#list of dict{'question_id': que_id,'answer': model's answer} (batch x batch_size) x dict
#val2014
def get_eval(result, split):
    if split not in ('train2014', 'val2014'):
        raise ValueError('split must be "train2014" or "val2014"')
    vqa_tool = VQATool()
    vqa = vqa_tool.get_vqa(cfg.DATA_DIR, split)
    que_fname = vqa_tool.get_que_path(cfg.DATA_DIR, split)

    res_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump(result, res_file)
    res_file.close()

    vqa_res = vqa.loadRes(res_file.name, que_fname)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)

    vqa_eval.evaluate()

    os.unlink(res_file.name)
    return vqa_eval

