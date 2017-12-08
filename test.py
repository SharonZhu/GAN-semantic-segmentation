# -*- coding: utf-8 -*-
# @Time     : 2017/12/8  下午2:57
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

from scipy import misc
gtFile/train/aachen/aachen_000000_000019_gtFine_labelIds.png
path = 'data/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
image = misc.imread(path)
print(image)