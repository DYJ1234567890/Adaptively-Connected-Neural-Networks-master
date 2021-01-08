#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py
# Author: Guangrun Wang

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    AvgPooling, Conv2D, GlobalAvgPooling, FullyConnected,
    LinearWrap, BNReLU, BatchNorm)

from tensorpack.models.common import layer_register

import cv2

import numpy as np

import random
import math

from tensorpack.tfutils.tower import get_current_tower_context

from gr_conv import Grconv

# 本函数是为了使shortcut的输入输出的通道数相同
def resnet_shortcut(l, n_out, stride, activation=tf.identity):   # tf.identity 返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作
    data_format = get_arg_scope()['Conv2D']['data_format']
    # 下面这个式子主要是为了提取出l的通道数
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3] # NCHW即 N,channel,height,width,也就是channels_first
    if n_in != n_out:   # 通道不同时改变尺寸，使之与n_out相同
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l

# 预激活 使实验效果更好
def apply_preactivation(l, preact):
    # bn批量正则化
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping  保留标识映射
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut

# 由于ResNet的结构有别于传统的卷积结构，使得信号的前向传播和梯度的反向传播变得更复杂。
# 为了稳定训练时信号的前向传播和梯度的反向传播，从ResNet开始，网络普遍使用Batch Normalization。

# 是否进行0初始化
def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)

# BasicBlock，它由两个 （3, 3, out_plane）的Conv2d 堆叠而成。在使用这个BasicBlock时候，只需要根据 堆叠具体参数：
# 输入输出通道数目，堆叠几个BasicBlock，就能确定每个stage中basicblock的基本使用情况


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact) # 先预激活
    # l是输入 ch_out是卷积核，3是kernel size
    l = Grconv('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Grconv('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)

# bottleneck先通过一个1x1的卷积减少通道数，使得中间卷积的通道数减少为1/4；中间的普通卷积做完卷积后输出通道数等于输入通道数；
# 第三个卷积用于增加（恢复）通道数，使得bottleneck的输出通道数等于bottleneck的输入通道数。这两个1x1卷积有效地较少了卷积的
# 参数个数和计算量。


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)  # 先预激活
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Grconv('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    # l是F(x),shortcut是x！！！！！！！！！！！！！！！！
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)  # l 和shortcut可以相加的条件是他们的维度和通道数相同，让shortcut和ch*4相同


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):  # 指定作用域
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation  第一个块不需要激活
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Grconv('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Grconv('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Grconv('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))

# se-resnet(改进了)
def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3   # ch_ax是data_format中表示通道数的索引值
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4   # 改变shape中通道数的值
    l = l * tf.reshape(squeeze, shape)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))

def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
                # l = BNReLU('block{}'.format(i), l)
    return l

# 残差网络的几个骨干步骤
def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope([Conv2D, Grconv], use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # LinearWrap创建线性图
        logits = (LinearWrap(image).Conv2D('conv0', 64, 7, strides=2, activation=BNReLU)())
        logits = (LinearWrap(logits).MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)())
        logits = (LinearWrap(logits).GlobalAvgPooling('gap')())
        logits = (LinearWrap(logits).FullyConnected('linear', 1000)())
    return logits
