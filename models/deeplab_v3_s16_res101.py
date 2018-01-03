# -*- coding: utf-8 -*-
# @Time     : 2017/12/7  下午3:43
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : deeplab_v3_s16_res101.py
# @Software : PyCharm

import sys
import tensorflow as tf
from models.network import Network

class DeepLabV3_101(Network):
    def setup(self, is_training, num_classes):
        inputs = self.inputs.popitem()[0]

        (self.feed(inputs)
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res6a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6a_branch2a')
             .atrous_conv([3, 3], 512, 4, padding='SAME', biased=False, relu=False, name='fc_res6a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6a_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res6a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn6a_branch2c'))

        (self.feed(inputs,
                   'fc_bn6a_branch2c')
             .add(name='fc_res6a')
             .relu(name='fc_res6a_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res6b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6b_branch2a')
             .atrous_conv([3, 3], 512, 8, padding='SAME', biased=False, relu=False, name='fc_res6b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6b_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res6b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn6b_branch2c'))

        (self.feed('fc_res6a_relu',
                   'fc_bn6b_branch2c')
             .add(name='fc_res6b')
             .relu(name='fc_res6b_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res6c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6c_branch2a')
             .atrous_conv([3, 3], 512, 16, padding='SAME', biased=False, relu=False, name='fc_res6c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn6c_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res6c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn6c_branch2c'))

        (self.feed('fc_res6b_relu',
                   'fc_bn6c_branch2c')
             .add(name='fc_res6c')
             .relu(name='fc_res6c_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res7a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7a_branch2a')
             .atrous_conv([3, 3], 512, 8, padding='SAME', biased=False, relu=False, name='fc_res7a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7a_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res7a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn7a_branch2c'))

        (self.feed('fc_res6c_relu',
                   'fc_bn7a_branch2c')
             .add(name='fc_res7a')
             .relu(name='fc_res7a_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res7b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7b_branch2a')
             .atrous_conv([3, 3], 512, 16, padding='SAME', biased=False, relu=False, name='fc_res7b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7b_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res7b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn7b_branch2c'))

        (self.feed('fc_res7a_relu',
                   'fc_bn7b_branch2c')
             .add(name='fc_res7b')
             .relu(name='fc_res7b_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res7c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7c_branch2a')
             .atrous_conv([3, 3], 512, 32, padding='SAME', biased=False, relu=False, name='fc_res7c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn7c_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res7c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn7c_branch2c'))

        (self.feed('fc_res7b_relu',
                   'fc_bn7c_branch2c')
             .add(name='fc_res7c')
             .relu(name='fc_res7c_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res8a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8a_branch2a')
             .atrous_conv([3, 3], 512, 16, padding='SAME', biased=False, relu=False, name='fc_res8a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8a_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res8a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn8a_branch2c'))

        (self.feed('fc_res7c_relu',
                   'fc_bn8a_branch2c')
             .add(name='fc_res8a')
             .relu(name='fc_res8a_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res8b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8b_branch2a')
             .atrous_conv([3, 3], 512, 32, padding='SAME', biased=False, relu=False, name='fc_res8b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8b_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res8b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn8b_branch2c'))

        (self.feed('fc_res8a_relu',
                   'fc_bn8b_branch2c')
             .add(name='fc_res8b')
             .relu(name='fc_res8b_relu')
             .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='fc_res8c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8c_branch2a')
             .atrous_conv([3, 3], 512, 64, padding='SAME', biased=False, relu=False, name='fc_res8c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='fc_bn8c_branch2b')
             .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='fc_res8c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc_bn8c_branch2c'))

        (self.feed('fc_res8b_relu',
                   'fc_bn8c_branch2c')
             .add(name='fc_res8c')
             .relu(name='fc_res8c_relu'))

        (self.feed('fc_res8c_relu')
             .atrous_conv([3, 3], 256, 6, padding='SAME', relu=False, name='fc1_voc12_c0')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn0'))

        (self.feed('fc_res8c_relu')
             .atrous_conv([3, 3], 256, 12, padding='SAME', relu=False, name='fc1_voc12_c1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn1'))

        (self.feed('fc_res8c_relu')
             .atrous_conv([3, 3], 256, 18, padding='SAME', relu=False, name='fc1_voc12_c2')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn2'))

        (self.feed('fc_res8c_relu')
             .conv([1, 1], 256, [1, 1], relu=False, name='fc1_voc12_c3')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn3'))

        layer = self.get_appointed_layer('fc_res8c_relu')
        new_shape = tf.shape(layer)[1:3]

        (self.feed('fc_res8c_relu')
             .global_average_pooling(name='fc1_voc12_mp0')
             .conv([1, 1], 256, [1, 1], relu=False, name='fc1_voc12_c4')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn4')
             .resize(new_shape, name='fc1_voc12_bu0'))

        (self.feed('fc1_voc12_bn0',
                   'fc1_voc12_bn1',
                   'fc1_voc12_bn2',
                   'fc1_voc12_bn3',
                   'fc1_voc12_bu0')
             .concat(axis=3, name='fc1_voc12'))

        (self.feed('fc1_voc12')
             .conv([1, 1], 256, [1, 1], relu=False, name='fc2_voc12_c0')
             .batch_normalization(is_training=is_training, activation_fn=None, name='fc2_voc12_bn0')
             .conv([1, 1], num_classes, [1, 1], relu=False, name='fc2_voc12_c1'))

    def topredict(self, raw_output, origin_shape):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, dimension=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction
