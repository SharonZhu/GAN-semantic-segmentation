# -*- coding: utf-8 -*-
# @Time     : 2017/12/6  下午5:07
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : make_txt.py
# @Software : PyCharm

import os
import sys
import glob
DATA_DIR = './data/leftImg8bit/test/'

def write_line(file, image_dir, segmentation_dir):
    file.write(image_dir + ' ')
    file.write(segmentation_dir)
    file.write('\n')

def making_txt(set, data_path, names):
    txt_file = open('./data/dataset/' + set + '_city.txt', 'w')

    for name in names:
        path = os.path.join(data_path, name, '*g')
        files = glob.glob(path)

        for f in files:
            print(f)
            img_dir = f.split('data/')[1]
            seg_dir = 'gtFine' + img_dir.split('leftImg8bit')[1] + 'gtFine_labelTrainIds.png'
            print(img_dir)
            print(seg_dir)
            write_line(txt_file, img_dir, seg_dir)
    txt_file.close()


if __name__ == '__main__':
    names_train = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt',
             'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 'strasbourg',
             'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    names_val = ['frankfurt', 'lindau', 'munster']
    names_test = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']
    making_txt('test', DATA_DIR, names_test)