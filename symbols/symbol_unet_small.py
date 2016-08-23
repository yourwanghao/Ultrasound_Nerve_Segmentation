import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
import cv2
import os

import sys
sys.path.append("..")
from symbols.symbol_common import *

def convolution_module(net, kernel_size=(3,3), pad_size=(1,1), stride=(1, 1), filter_count = 32, work_space=2048, batch_norm=True, stage="stage1", bsize=settings.SEGMENT_BATCH_SIZE):
    net = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    if (batch_norm):
        net = mx.sym.BatchNorm(net)
    #net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.LeakyReLU(net, act_type='elu')
    print_inferred_shape(net, stage=stage, bsize=bsize)

    net = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    if (batch_norm):
        net = mx.sym.BatchNorm(net)
    #net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.LeakyReLU(net, act_type='elu')
    print_inferred_shape(net, stage=stage, bsize=bsize)

    return net

def upconvolution_module(net, kernel_size=(2,2), pad_size=(0,0), stride_size=(2, 2), filter_count = 32, work_space=2048, batch_norm=True, stage="stage1", bsize=settings.SEGMENT_BATCH_SIZE):
    net = mx.sym.Deconvolution(net, kernel=kernel_size, pad=pad_size, stride=stride_size, num_filter=filter_count, workspace = work_space)
    if (batch_norm):
        net = mx.sym.BatchNorm(net)
    #net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.LeakyReLU(net, act_type='elu')
    print_inferred_shape(net, stage=stage, bsize=bsize)

    return net

def downpool(net, kernel = (2, 2), stride = (2, 2), pad = (0, 0), stage="stage1", bsize=settings.SEGMENT_BATCH_SIZE):

    net = mx.symbol.Pooling(net, pool_type="max", kernel = kernel, stride = stride, pad = pad)
    print_inferred_shape(net, stage=stage, bsize=bsize)
    return net

def get_unet_small(batch_size=settings.SEGMENT_BATCH_SIZE):
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    conv_pad_size = (1, 1)
    pad_size = (1, 1)
    stride_size = (1, 1)
    filter_count = 32

    conv1 = convolution_module(source, kernel_size, pad_size=(1,1), stride=stride_size, filter_count=filter_count, stage="stage1", bsize=batch_size)
    net = conv1
    net = downpool(net, stage="stage1", bsize=batch_size)

    conv2 = convolution_module(net, kernel_size, pad_size=(1,1) , stride = stride_size, filter_count=2*filter_count, stage="stage2", bsize=batch_size)
    net = conv2
    net = downpool(net, stage="stage2", bsize=batch_size)

    conv3 = convolution_module(net, kernel_size, pad_size, stride_size, filter_count=4*filter_count, stage="stage3", bsize=batch_size)
    net = conv3
    net = downpool(net, stage="stage3", bsize=batch_size)

    conv4 = convolution_module(net, kernel_size, pad_size, stride_size, filter_count=8*filter_count, stage="stage4", bsize=batch_size)
    net = conv4
    net = downpool(net, stage="stage4", bsize=batch_size)

    conv5 = convolution_module(net, kernel_size, pad_size, stride_size, filter_count=16*filter_count, stage="stage5", bsize=batch_size)
    net = conv5

    upconv6 = upconvolution_module(net, stage="stage6", filter_count=8*filter_count, bsize=batch_size)
    net = mx.sym.Concat(*[conv4, upconv6])
    conv7 = convolution_module(net, kernel_size, pad_size, stride_size, filter_count=8*filter_count, stage="stage7", bsize=batch_size)
    net = conv7

    upconv8 = upconvolution_module(net, stage="stage8", filter_count=4*filter_count, bsize=batch_size)
    net = mx.sym.Concat(*[conv3, upconv8])
    conv9 = convolution_module(net, kernel_size, pad_size, stride_size, filter_count=4*filter_count, stage="stage9", bsize=batch_size)
    net = conv9

    upconv10 = upconvolution_module(net, stage="stage10", filter_count=2*filter_count, bsize=batch_size)
    net = mx.sym.Concat(*[conv2, upconv10])
    conv11 = convolution_module(net, kernel_size, pad_size=(1,1), stride=(1,1), filter_count=2*filter_count, stage="stage11", bsize=batch_size)
    net = conv11

    upconv12 = upconvolution_module(net, pad_size=(0,0), stage="stage12", filter_count=filter_count, bsize=batch_size)
    net = mx.sym.Concat(*[conv1, upconv12])
    conv13 = convolution_module(net, kernel_size, pad_size=(1,1), stride=stride_size, filter_count=filter_count, stage="stage13", bsize=batch_size)
    net = conv13


    final_filter =  2
    final_kernel = (3, 3)
    final_pad = (1, 1)
    final_stride=(1, 1)
    net = mx.sym.Convolution(data=net, kernel=final_kernel, stride=final_stride, pad=final_pad, num_filter=final_filter,
                              workspace=1024)
    net = mx.sym.Reshape(net, shape=(batch_size, final_filter, settings.SCALE_WIDTH * settings.SCALE_HEIGHT))
    print_inferred_shape(net, stage="stage14", bsize=batch_size)

    net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)

    #net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    return net

if __name__ == "__main__":
    get_unet_small()
