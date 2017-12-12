# -*- coding: utf-8 -*-
# @Time     : 2017/12/8  下午2:57
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from utils import decode_labels
import sys

image_path = 'data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
image = misc.imread(image_path)
print(image)
plt.figure(1)
plt.imshow(image)

label_path = 'data/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
label = misc.imread(label_path)
plt.figure(2)
plt.imshow(label, cmap='gray')

print(label)

label = np.reshape(label, [-1, 1024, 2048, 1])
mask = decode_labels(label)
print(mask)
print(np.where(mask!=0))
plt.figure(3)
mask = np.reshape(mask, [1024, 2048, 3])
plt.imshow(mask)
plt.show()