import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
import cv2
import os

from symbol_common import *

def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True,
                       down_pool=False, up_pool=False, act_type="relu", convolution=True, stage="stage1"):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                                   workspace=work_space)
        net = mx.sym.BatchNorm(net)
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type)
        print_inferred_shape(net, stage=stage, bsize=settings.SEGMENT_BATCH_SIZE)

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                                  workspace=work_space)
        net = conv
        print_inferred_shape(conv, stage=stage, bsize=settings.SEGMENT_BATCH_SIZE)

    if batch_norm:
        net = mx.sym.BatchNorm(net)

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type)

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        net = pool
        print_inferred_shape(net, stage=stage, bsize=settings.SEGMENT_BATCH_SIZE)

    return net


def get_unet_orig():
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    pad_size = (1, 1)
    filter_count = 32
    pool1 = convolution_module(source, kernel_size, pad_size=(1,1), filter_count=filter_count, down_pool=True, stage="stage1")
    net = pool1
    pool2 = convolution_module(net, kernel_size, pad_size=(1, 1), filter_count=filter_count * 2, down_pool=True, stage="stage2")
    net = pool2
    pool3 = convolution_module(net, kernel_size, pad_size=(1, 1), filter_count=filter_count * 4, down_pool=True, stage="stage3")
    net = pool3
    pool4 = convolution_module(net, kernel_size, pad_size=(1, 1), filter_count=filter_count * 4, down_pool=True, stage="stage4")
    net = pool4
    net = mx.sym.Dropout(net)
    pool5 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 8, down_pool=True, stage="stage5")
    net = pool5
    net = convolution_module(net, kernel_size, pad_size=(1,1), filter_count=filter_count * 4, up_pool=True, stage="stage6")
    net = convolution_module(net, kernel_size, pad_size=(1,1), filter_count=filter_count * 4, up_pool=True, stage="stage7")

    # dirty "CROP" to wanted size... I was on old MxNet branch so user conv instead of crop for cropping
    #net = convolution_module(net, (4, 4), (0, 0), filter_count=filter_count * 4, stage="stage8")
    net = mx.sym.Concat(*[pool3, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, stage="stage9")
    net = convolution_module(net, kernel_size, pad_size=(1,1), filter_count=filter_count * 4, up_pool=True, stage="stage10")

    net = mx.sym.Concat(*[pool2, net])
    print_inferred_shape(net, stage="stage10", bsize=settings.SEGMENT_BATCH_SIZE)
    #net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, stage="stage11")
    net = convolution_module(net, kernel_size, pad_size=(1,1), filter_count=filter_count * 4, up_pool=True, stage="stage12")
    #convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool1, net])
    #net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size=(1,1), filter_count=filter_count * 2, stage="stage13")
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, up_pool=True, stage="stage14")

    net = convolution_module(net, kernel_size, pad_size, filter_count=2, batch_norm=False, act_type="", stage="stage15")

    #net = mx.symbol.Flatten(net)
    net = mx.symbol.Reshape(net, shape=(settings.SEGMENT_BATCH_SIZE, 2, settings.SCALE_HEIGHT*settings.SCALE_WIDTH))
    print_inferred_shape(net, stage="stage16", bsize=settings.SEGMENT_BATCH_SIZE)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)
    #net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    return net


if __name__ == "__main__":
    get_unet_orig()
