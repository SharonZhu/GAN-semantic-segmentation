# -*- coding: utf-8 -*-
# @Time     : 2017/12/12  下午3:12
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : make_labelID.py
# @Software : PyCharm

import os
import sys
import glob

from PIL import Image
from scipy import misc
import numpy as np

# label_list = '/data/rui.wu/irfan/gan_seg/DAG4Seg/D_deeplab/dataset/val_label.txt'
id_to_trainid = {-1: 255, 0: 255, 1: 255, 2: 255,
3: 255, 4: 255, 5: 255, 6: 255,
7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
14: 255, 15: 255, 16: 255, 17: 5,
18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}
# label_dir = '/data/rui.wu/Elijha/dataset/gtFine/gtFine/val/munster/munster_000100_000019_gtFine_labelIds.png'
RAW_LABEL_DIR = '../data/gtFine/test/'

def make_labelID(labels_dir, names):
    for name in names:
        path = os.path.join(labels_dir, name, '*labelIds.*g')
        files = glob.glob(path)

        for f in files:
            print(f)
            label_dir = f

            # modify mask
            mask = misc.imread(label_dir)
            mask_copy = mask.copy()

            for k, v in id_to_trainid.items():
                mask_copy[mask == k] = v
            mask = Image.fromarray(mask_copy.astype(np.uint8))
            print('done:' + label_dir)
            mask.save(label_dir.replace('labelIds', 'labelTrainIds'))


if __name__ == '__main__':
    names_train = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt',
                   'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 'strasbourg',
                   'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    names_val = ['frankfurt', 'lindau', 'munster']
    names_test = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']
    make_labelID(RAW_LABEL_DIR, names_test)


