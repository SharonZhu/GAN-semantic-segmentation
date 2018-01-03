# -*- coding: utf-8 -*-
# @Time     : 2018/1/2  下午5:51
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : train_v3.py
# @Software : PyCharm

"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import time

from config import *
from datetime import datetime
from utils import decode_labels, inv_preprocess, prepare_label
from image_reader import ImageReader
from models.nets.deeplabv3 import deeplabv3

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou


def save(saver, sess, logdir, step):
    '''Save weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    print(ckpt_path)
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("successful loading,global step is %s" % global_step)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def main():
    """Create the model and start the training."""
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        print(image_batch, label_batch)
    # Create network.
    net, end_points = deeplabv3(image_batch,
                                num_classes=args.num_classes,
                                depth=args.num_layers,
                                is_training=True)

    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = end_points['resnet{}/logits'.format(args.num_layers)]
    print(raw_output)
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name
                   or not args.not_restore_last]
    if args.freeze_bn:
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in
                         v.name and 'gamma' not in v.name]
    else:
        all_trainable = [v for v in tf.trainable_variables()]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]
    # Upsample the logits instead of donwsample the ground truth
    raw_output_up = tf.image.resize_bilinear(raw_output, [h, w])

    # Predictions: ignoring all predictions with labels greater or equal than
    # n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    # label_proc = tf.squeeze(label_batch)
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes,
                               one_hot=False)  # [batch_size, h, w]
    print(label_proc)
    # mask = label_proc <= args.num_classes
    # seg_logits = tf.boolean_mask(raw_output_up, mask)
    # seg_gt = tf.boolean_mask(label_proc, mask)
    # seg_gt = tf.cast(seg_gt, tf.int32)
    # print(seg_gt)
    # print(seg_logits)
    # sys.exit()
    raw_gt = tf.reshape(label_proc, [-1, ])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    seg_gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    print(prediction)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                          labels=seg_gt)
    seg_loss = tf.reduce_mean(loss)
    # seg_loss_sum = tf.summary.scalar('loss/seg', seg_loss)
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_loss = tf.add_n(reg_losses)
    # reg_loss_sum = tf.summary.scalar('loss/reg', reg_loss)
    # tot_loss = seg_loss + reg_loss
    # tot_loss_sum = tf.summary.scalar('loss/tot', tot_loss)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    seg_pred = tf.expand_dims(raw_output_up, dim=3)
    print(seg_pred)

    # train_mean_iou, train_update_mean_iou = streaming_mean_iou(seg_pred,
    #                                                            seg_gt, args.num_classes, name="train_iou")
    # train_iou_sum = tf.summary.scalar('accuracy/train_mean_iou',
    #                                   train_mean_iou)
    train_initializer = tf.variables_initializer(var_list=tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="train_iou"))

    print(args.num_classes)
    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [seg_pred, args.save_num_images, args.num_classes], tf.uint8)

    image_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                     max_outputs=args.save_num_images)  # Concatenate row-wise.


    total_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    # opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    # opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable)

    train_op = slim.learning.create_train_op(
        reduced_loss, opt,
        global_step=global_step,
        variables_to_train=conv_trainable,
        summarize_gradients=True)


    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        if args.restore_from is not None:
            loader = tf.train.Saver()
            load(loader, sess, args.restore_from)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        # train_op = adam

        # Iterate over training steps.
        for step in range(args.num_steps):
            start_time = time.time()
            feed_dict = {step_ph: step}

            if step % args.save_pred_every == 0:
                loss_value, images, labels, preds, summary, _ = sess.run(
                    [reduced_loss, image_batch, label_batch, seg_pred, total_summary, train_op], feed_dict=feed_dict)
                # train_mean_iou_float = sess.run(train_mean_iou)
                duration = time.time() - start_time
                sys.stdout.write('step {:d}, tot_loss = {:.6f}' \
                                 'sec/step)\n'.format(step, loss_value, duration)
                                 )
                sys.stdout.flush()

                # print(images.shape)
                # print(labels.shape)
                # print(labels[0].shape)
                # plt.imshow(images[0])
                # plt.imshow(labels[0], cmap='gray')
                # plt.show()
                # sys.exit()

                summary_writer.add_summary(summary, step)
                save(saver, sess, args.snapshot_dir, step)
            else:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
