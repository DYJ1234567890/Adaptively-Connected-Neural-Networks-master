#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: p_norm.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)


import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from tensorpack.utils import logger
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack import *
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format

from tensorpack.models import (BNReLU)


import numpy as np

__all__ = ['Grconv']

def fc(x, out_shape,padding='same',
    data_format='channels_first',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    split=1,): 
    x = GlobalAvgPooling('gap', x)      #全局池化
    x = FullyConnected('fc1', x, out_shape[1], activation=tf.nn.relu)
    x = FullyConnected('fc3', x, out_shape[1])
    x = tf.reshape(x, [-1, out_shape[1], 1, 1])
    return x


@layer_register()
def Grconv(x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='same',
    data_format='channels_last',   #'channels_last'默认 表示输入中维度的顺序。 channels_last 对应输入尺寸为 (batch, height, width, channels)， channels_first 对应输入尺寸为 (batch, channels, height, width)。
    dilation_rate=(1, 1), # 一个整数或 2 个整数的元组或列表,为所有空间维度指定相同的值。 当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。
    activation=None,
    use_bias=True, # 布尔值，该层是否使用偏置向量。
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0), # 一种更好的方差初始化的方法，调整初始随机权重的方差
    bias_initializer=tf.zeros_initializer(),     #初始化为0
    kernel_regularizer=None, # 运用到 kernel 权值矩阵的正则化函数
    bias_regularizer=None, #  运用到偏置向量的正则化函数
    activity_regularizer=None, # 运用到层输出（它的激活值）的正则化函数
    split=1,
    glocal = False):
    z3 = Conv2D('z3', x, filters=filters, kernel_size=3, strides=strides, 
        padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=tf.identity, use_bias=use_bias, 
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, split=1)  # tf.identity 线性操作
    out_shape = z3.get_shape().as_list()  # 输出z3的维度并以列表的方式显示

    z1 = Conv2D('z1', x, filters=filters, kernel_size=1, strides=strides, 
        padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=tf.identity, use_bias=use_bias, 
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, split=1)
    zf = fc(x, out_shape,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=tf.identity,
        use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer)  # zf为经过池化及全连接后的网络

    p = tf.get_variable('PPP', [3, 1, out_shape[2], out_shape[3]], initializer=tf.ones_initializer(), trainable = True)
    # 创建一个名为'ppp' ,形状为，用initializer初始化的变量，并将变量添加到图形集合（trainable）
    p = tf.nn.softmax(p, 0)

    z = p[0:1,:,:,:] * z3 + p[1:2,:,:,:] * z1 + p[2:3,:,:,:] * zf # 存疑 三个样本
    z = BNReLU('sum', z) #这是一个其他函数
    return z