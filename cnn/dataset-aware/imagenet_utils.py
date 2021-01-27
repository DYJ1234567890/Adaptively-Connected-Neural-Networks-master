#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)

from tensorpack import *
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.tfutils.tower import get_current_tower_context

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.             对原始图像进行处理
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224): # crop_area_fraction裁剪区域比例
        self._init(locals())                                   # 返回包含当前作用域的局部变量的字典。
        #self.crop_area_fraction = crop_area_fraction           # 此句是自己加的
        #self.aspect_ratio_low = aspect_ratio_low               # 此句是自己加的
        #self.aspect_ratio_high = aspect_ratio_high             # 此句是自己加的
        #self.target_shape = target_shape                       # 此句是自己加的
    # 本函数的功能应该就是把图像扩展到要求的维度

    def _augment(self, img, _):
        h, w = img.shape[:2]     #获取图像的行数和列数
        area = h * w             #相当于28*28 要处理的整个区域的大小
        for _ in range(10):      # 循环十次
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area          # 确定目标区域 随机数生成器
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)   # 随机设置 0.75~1.333的数字
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)          # 上面四步是确定要处理的目标区域的长和宽
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww                                    # 如果随机数小于0.5 说明...互换
            if hh <= h and ww <= w:                                # 如果处理的区域小于总长度 正常处理 否则开始下一次循环
                # x1，y1为起始点坐标，out是确定的随机区域
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out

# 根据是否训练产生解析器参数列表
def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255]. # 残差网络增强图像
    """
    if isTrain:         # 如果训练数据的话
        augmentors = [
            GoogleNetResize(),  # 定义好了crop_area_fraction等参数
            imgaug.RandomOrderAug(      # GPU不行的话就把这部分删除Remove these augs if your CPU is not fast enough #imgaug是一个图像增强库
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:  # 如果不是训练数据的话
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),     #  在保持纵横比的同时，将最短边的大小调整为某个数字。
            imgaug.CenterCrop((224, 224)),  # 在中间裁剪图像
        ]
    return augmentors

# 本函数：根据是否为训练数据对数据进行不同的处理并获得数据流
# datadir:数据集的名字
# name：训练数据还是测试数据
def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors, parallel=None): #获取图像网络数据流
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:   # 如果不是并行的话
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading 超线程？ 获取当前计算机cpu数量

    if isTrain:
        # dataset:创建一个在数据流上运行的预测器，并且拿出一个batch？
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False) # 使用共享的增强参数在多个组件上应用图像增强器
        if parallel < 16:   # 如果少于16个的话
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel) # 实现高效的数据流水线
        ds = BatchData(ds, batch_size, remainder=False)  # 取一个batch？
    else:
        # 如果是测试时,增强图像，加速对数据流的读取操作等
        # 与ILSVRC12相同，但生成图像的文件名而不是np array。
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR : 默认使用该种标识。加载一张彩色图片,忽视它的透明度
            im = aug.augment(im)     # 增强图像
            return im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)  # 并行加速？
        ds = BatchData(ds, batch_size, remainder=True)    # 取一个batch?
        ds = PrefetchDataZMQ(ds, 1)
    return ds

# 本函数用于评估预测结果，输出误差
def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    ) # 该函数用于构件图
    pred = SimpleDatasetPredictor(pred_config, dataflow)  # Simply create one predictor and run it on the DataFlow.
    acc1, acc5 = RatioCounter(), RatioCounter()   #  A counter to count ratio of something.某事物的记录
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))  # 输出误差


class ImageNetModel(ModelDesc):
    weight_decay = 1e-4
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Whether to apply weight decay on BN parameters.是否在BN层用权值衰减
    """
    weight_decay_on_bn = False

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'
    # 对标签和输入占位
    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]   # 输入和标签占位

    # 本函数：根据是否权值衰减求出总的cost
    # 本函数与ghost的不同之处就在于本函数在wd_loss之前对wd_w进行了权值指数衰减
    def build_graph(self, image, label):  # 本函数返回总的损失
        image = ImageNetModel.image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)

        # ctx = get_current_tower_context()
        # is_training = ctx.is_training
        # if is_training:
        #     label = tf.concat([label, label, label[0:17]], axis = 0)
            
        loss = ImageNetModel.compute_loss_and_error(logits, label)

        if self.weight_decay > 0:   # 如果权值衰减大于0
            if self.weight_decay_on_bn:  # 如果在bn上也进行权值衰减的话
                pattern = '.*/W|.*/gamma|.*/beta'  # 匹配变量名的正则表达式
            else:
                pattern = '.*/W'

            wd_w = tf.train.exponential_decay(self.weight_decay, get_global_step_var(),
                                          150000, 1.2, True)  # 指数衰减来解决学习率的问题
            wd_loss = regularize_cost(pattern, tf.contrib.layers.l2_regularizer(wd_w),
                                      name='l2_regularize_loss')  #返回总的正则化后的loss 标量
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)
        return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Nx1000 logits
        """
    # 优化 学习率？
    def optimizer(self):  # 做优化的
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)  # 学习率0.1
        tf.summary.scalar('learning_rate-summary', lr)  # 收集一维标量，一般在画loss,accuary时会用到这个函数
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    @staticmethod
    # 图像预处理
    def image_preprocess(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)     # 归一化？

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    # 计算误差和损失 以及错误率
    def compute_loss_and_error(logits, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss

# 这个是用来测试在val和train下数据流的速度
if __name__ == '__main__':
    import argparse
    from tensorpack.dataflow import TestDataSpeed
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--aug', choices=['train', 'val'], default='train')
    args = parser.parse_args(['--data', 'C:\\Users\\丁丁学长\\Desktop\\ImageNet-ResNet50.npz'])
    # 根据是否训练产生参数列表
    if args.aug == 'val':
        augs = fbresnet_augmentor(False)
    elif args.aug == 'train':
        augs = fbresnet_augmentor(True)
    df = get_imagenet_dataflow(
        args.data, 'train', args.batch, augs)
    # For val augmentor, Should get >100 it/s (i.e. 3k im/s) here on a decent E5 server.
    TestDataSpeed(df).start()