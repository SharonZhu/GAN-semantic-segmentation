# _*_ coding:utf-8
import time

import numpy as np
import tensorflow as tf

from utils import prepare_label, inv_preprocess, decode_labels
from image_reader import read_labeled_image_list
from train import load as load_weight


def convert_to_calculateloss(raw_output, label_batch, num_classes):
    #label_proc = prepare_label(label_batch, tf.shape(raw_output)[1:3],
    #                           num_classes=num_classes, one_hot=False)  # [batch_size, h, w]
    raw_groundtruth = tf.reshape(tf.squeeze(label_batch, squeeze_dims=[3]), [-1, ])
    raw_prediction = tf.reshape(raw_output, [-1, num_classes])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_groundtruth, num_classes - 1)), 1)
    label = tf.cast(tf.gather(raw_groundtruth, indices), tf.int32)  # [?, ]
    logits = tf.gather(raw_prediction, indices)  # [?, num_classes]

    return label, logits


def get_validate_data(image_name, label_name, img_mean):
    img = tf.read_file(image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    img -= img_mean
    img = tf.expand_dims(img, axis=0)

    label = tf.read_file(label_name)
    label = tf.image.decode_png(label, channels=1)
    label = tf.expand_dims(label, axis=0)

    return img, label


def image_cropping(img,crop_h=513,crop_w=513):
    img_shape = tf.shape(img)
    combined_pad = tf.image.pad_to_bounding_box(img, 0, 0, tf.maximum(crop_h, img_shape[0]),
                                                tf.maximum(crop_w, img_shape[1]))
    last_image_dim = tf.shape(img)[-1]
    img_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 3])
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop

def image_scaling(img, scales):

    imgs = []
    original_shape = tf.shape(imgs)
    for scale in scales:
        h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
        w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[2]), scale))
        new_shape = tf.to_int32(tf.stack([h_new, w_new]))
        new_img = tf.cast(tf.image.resize_images(img, new_shape), dtype=tf.float32)
        imgs.append(new_img)
    return imgs

def val(args, dbargs):

    if args.val_on_16:
        print('import net_s16')
        from models.deeplabnet_s16 import DeepLabV2, DeepLabV3
        from models.resnet_s16 import ResNet38, ResNet101
    elif args.val_on_4:
        print('import net_s4')
        from models.deeplabnet_s4 import DeepLabV2, DeepLabV3
        from models.resnet_s4 import ResNet38, ResNet101
    else:
        print('import net_s8')
        from models.deeplabnet_s8 import DeepLabV2, DeepLabV3
        from models.resnet_s8 import ResNet38, ResNet101

    def choose_model(model_name, base_model, image_batch):
        net = None
        if base_model == 'resnet38':
            net = ResNet38(inputs={'data': image_batch}).terminals[-1]
        elif base_model == 'resnet101':
            net = ResNet101(inputs={'data': image_batch}).terminals[-1]
        if model_name == 'deeplabv2':
            net = DeepLabV2(inputs={net.op.name: net})
        elif model_name == 'deeplabv3':
            net = DeepLabV3(inputs={net.op.name: net})

        return net

    ## set hyparameter
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    scales = [1.25]
    if args.multi_scale:
        scales = [0.5,0.75,1.0,1.25,1.5,1.75]
    tf.set_random_seed(args.random_seed)
    scales = [s*513/500.0  for s in scales]
    print('scales: ' + str(scales))

    ## load data
    image_list, label_list, png_list = read_labeled_image_list(args.data_dir, is_val=True)
    num_val = len(image_list)
    image_name = tf.placeholder(dtype=tf.string)
    label_name = tf.placeholder(dtype=tf.string)
    png_name = tf.placeholder(dtype=tf.string)
    image_batch, label_batch = get_validate_data(image_name, label_name, img_mean)
    # image_batch_crop= image_cropping(image_batch)
    images_batch = image_scaling(image_batch, scales)

    print("data load completed!")
    ## load model
    raw_outputs = []

    for i, image in enumerate(images_batch):
      with tf.variable_scope('', reuse=False if i is 0 else True):
          net = choose_model(args.model_name, args.base_model, image)
      if not args.flip_mirror:
          raw_output = net.terminals[-1]
          raw_output = tf.cast(tf.image.resize_images(raw_output, tf.shape(label_batch)[1:3]), dtype=tf.float32)
      else:
          with tf.variable_scope('', reuse=True):
              print('Adding left-right flipped images during inference.')
              image_mirror = tf.reverse(image, [2])
              net_mirror = choose_model(args.model_name, args.base_model, image_mirror)
              raw1 = tf.cast(tf.image.resize_images(net.terminals[-1], tf.shape(label_batch)[1:3]), dtype=tf.float32)
              raw2 = tf.cast(tf.image.resize_images(tf.reverse(net_mirror.terminals[-1], [2]), tf.shape(label_batch)[1:3]), dtype=tf.float32)
              raw_output = tf.add_n([raw1, raw2])
      raw_outputs.append(raw_output)
    # ----------------
    raw_output_ph = tf.placeholder(shape=raw_output.get_shape().as_list(),dtype=tf.float32)
    predict_batch = net.topredict(raw_output_ph, tf.shape(label_batch)[1:3])
    predict_img = tf.write_file(png_name,
                                tf.image.encode_png(tf.cast(tf.squeeze(predict_batch, axis=0), dtype=tf.uint8)))

    labels, logits = convert_to_calculateloss(tf.image.resize_bilinear(raw_output_ph, tf.shape(label_batch)[1:3]), label_batch, args.num_classes)
    pre_labels = tf.argmax(logits, 1)

    print("Model load completed!")

    iou, iou_op = tf.metrics.mean_iou(labels, pre_labels, args.num_classes, name='iou')
    acc, acc_op = tf.metrics.accuracy(labels, pre_labels)
    m_op = tf.group(iou_op, acc_op)

    image = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, img_mean], tf.uint8)
    label = tf.py_func(decode_labels, [label_batch, ], tf.uint8)
    pred = tf.py_func(decode_labels, [predict_batch, ], tf.uint8)
    tf.summary.image(name='img_collection_val', tensor=tf.concat([image, label, pred], 2))
    tf.summary.scalar(name='iou_val', tensor=iou)
    tf.summary.scalar(name='acc_val', tensor=acc)
    sum_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter(dbargs['log_dir'], max_queue=20)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session()
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    saver = tf.train.Saver(var_list=tf.global_variables())
    _ = load_weight(dbargs['restore_from'], saver, sess)

    print("validation begining")
    cp_path = tf.train.latest_checkpoint(dbargs['restore_from'])

    f = open('val.out', 'a')
    it = time.time()
    for step in range(num_val):
        feed_dict = {image_name: image_list[step], label_name: label_list[step], png_name: png_list[step]}
        raw_outs=[]
        for raw_out in raw_outputs:
            raw_out_ = sess.run(raw_out,feed_dict)
            raw_outs.append(np.squeeze(raw_out_))

        final_raw_outs = np.sum(np.array(raw_outs),axis=0,keepdims=True)
        if args.save_pred_png:
            _ = sess.run([predict_img], {raw_output_ph:final_raw_outs, **feed_dict})
            if step % 100 == 0 or step == num_val-1:
                print('step:{}/{}'.format(step, num_val))
            continue
        else:
            _, _ = sess.run([predict_img, m_op], {raw_output_ph:final_raw_outs, **feed_dict})
        if step % 50 == 0 or step == num_val-1:
            summ = sess.run(sum_op, {raw_output_ph:final_raw_outs, **feed_dict})
            sum_writer.add_summary(summ, step)
            print("step:{}, iou:{}, time:{}".format(step, iou.eval(session=sess), time.time() - it))

    print("end......")

    if not args.save_pred_png:
        final_iou = 'Mean IoU: {:.4f}'.format(iou.eval(session=sess))
        if args.val_on_16:
            val_on = 16
        elif args.val_on_4:
            val_on = 4
        else:
            val_on = 8
        f.write('val_on: {}\t multi: {}\t flip: {}\t weights: {}\t mIOU: {:.4f}\n'.format(val_on, args.multi_scale, args.flip_mirror, cp_path, iou.eval(session=sess)))
        print(final_iou)
