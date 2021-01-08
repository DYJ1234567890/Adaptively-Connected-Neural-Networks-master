#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)

import argparse
import os
from tensorpack import QueueInput
from tensorpack.utils import logger
from tensorpack.models import *


from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu
from imagenet_utils import (fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,eval_on_ILSVRC12)
from resnet_model import (preresnet_group, preresnet_basicblock, preresnet_bottleneck,resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,resnet_backbone)

from gr_conv import Grconv

# 继承父类：ImageNetModel
# 主要就是生成一个残差网络
class Model(ImageNetModel):
    def __init__(self, depth, mode='resnet'):
        if mode == 'se':
            assert depth >= 50   # 如果是SE网络，那么深度必须大于等于50否则直接报错

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock  # 基本块，预激活的或者是激活的
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]  # 由mode（网络形式）确定bottleneck
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            74: ([3, 4, 16, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]  # 由深度确定个数（Res18，34，...，152）以及他们的维度和块的类型
    # 本函数：根据深度和类型生成一个残差网络
    def get_logits(self, image):
        with argscope([Conv2D, Grconv, MaxPooling,  AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)


# 根据是否为训练数据对数据进行不同的处理，返回数据流或流水线
def get_data(name, batch):
    isTrain = name == 'train'
    # 根据是否是训练数据获得对应的augmentors
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors) # 返回图像数据流？

# 开始学习
def get_config(model, fake=False):
    start_ = 0
    # nr_tower GPU的数量
    nr_tower = max(get_nr_gpu(), 1)
    assert args.batch % nr_tower == 0
    # 每块GPU上的batch
    batch = args.batch // nr_tower
    # 加载日志，在第几块GPU上运行，batch是多少。
    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    if fake:
        # FakeData参数说明
        # shape（列表）：列表/元组的列表。每个组件的形状。
        # size（int）：此数据流的大小。
        # random（bool）：每次迭代是否随机生成数据。              请注意，仅仅生成数据有时可能会非常耗时！
        # dtype（str或list）：数据类型为string，或数据类型列表。
        # 这里是使用fakedata测试或基准测试此模型
        dataset_train = FakeData(
            [[batch, 224, 224, 3], [batch]], 1000, random=False, dtype='uint8')
        callbacks = [] # 该语句可能是为了让没有训练数据的时候也能产生一些伪数据
    else:
        dataset_train = get_data('train', batch)
        dataset_val = get_data('val', batch)

        START_LR = 0.1   # 开始的学习率
        BASE_LR = START_LR * (args.batch / 256.0)    # 基础学习率？
        if start_ < 31:
            lr_setting =[(max(30-start_, 0) , BASE_LR * 1e-1), (60 - start_, BASE_LR * 1e-2),(
                90 - start_, BASE_LR * 1e-3), (105 - start_, BASE_LR * 1e-4)]
        elif start_ < 61:
            lr_setting =[(max(60 - start_, 0), BASE_LR * 1e-2),(
                90 - start_, BASE_LR * 1e-3), (105 - start_, BASE_LR * 1e-4)]
        elif start_ < 91:
            lr_setting =[(max(90 - start_, 0), BASE_LR * 1e-3), (105 - start_, BASE_LR * 1e-4)]
        else:
            print('not found learning rate setting!!!!!!!!!!!!!')
        # callback包含保存模型，估算时间，通过预定义的基于时间的计划设置超参数等
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),
            ScheduledHyperParamSetter(
                'learning_rate', lr_setting),
            # TensorPrinter(['tower1/group3/block2/conv2/Abs_0', 'tower1/group3/block2/conv2/Abs_1:0', 'tower1/group3/block2/conv2/Abs_2:0'])
        ]
        if BASE_LR > START_LR:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, START_LR), (5, BASE_LR)], interp='linear'))

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        # 根据GPU的数量取出数据流
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1280000 // args.batch,
        max_epoch=120,
    )   # 返回参数 模型，数据流，callbacks（存放模型，估算时间等），步长


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   # 创建解析器
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')   # 添加参数
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[18, 34, 50, 74, 101, 152])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch', default=256, type=int,
                        help='total batch size. 32 per GPU gives best accuracy, higher values should be similarly good')
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    parser.add_argument('--log_dir', type=str, default='')
    args = parser.parse_args()  # 解析参数
    #  默认是Namespace(batch=256, data=None, data_format='NCHW', depth=50, eval=False, fake=False, gpu=None, load=None, log_dir='', mode='resnet')
    # 有GPU就用os.environ获得系统gpu信息
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 根据深度和模式生成一个残差网络，# model的输出方式默认为 NCHW
    model = Model(args.depth, args.mode)   # 前面的class Model(ImageNetModel) def __init__(self, depth, mode='resnet')
    model.data_format = args.data_format
    # 如果是测试的话（eval），输出错误率
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)  # 输出错误率
    # 如果是训练的话,记录日志并开始学习
    else:
        if args.fake:   # 如果使用fakedata测试或基准测试此模型
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            log_foder = '/data0/wangguangrun/log_acnt/imagenet-resnet-%s' % (args.log_dir)
            logger.set_logger_dir(os.path.join(log_foder))   # 保存路径？
        config = get_config(model, fake=args.fake)    # 得到参数
        # 如果要加载模型的话
        if args.load:
            config.session_init = get_model_loader(args.load)
        # 所有GPU一起开始训练
        trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)   # 最终训练的时候用到了get_optimizer 和 build_graph
