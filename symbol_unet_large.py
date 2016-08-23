import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
import cv2
import os

from symbol_common import *

def convolution(net, kernel_size, stride_size, pad_size, num_filter, workspace = 1024):
    conv = mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)

def get_unet_large(bsize=settings.SEGMENT_BATCH_SIZE):
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    stride = (1, 1)
    filter_count = 32
    work_space = 1024
    act_type="relu"

    #############################################################################################################
    stage = "down_stage1"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    relu1 = net
    print_inferred_shape(net, stage, bsize)

    pad_size = (0, 0)
    pool1 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), pad=pad_size)
    net = pool1
    print_inferred_shape(net, stage, bsize)


    #############################################################################################################
    stage = "down_stage2"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    relu2 = net
    print_inferred_shape(net, stage, bsize)

    pool2 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool2
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage3"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    relu3 = net
    print_inferred_shape(net, stage, bsize)

    pool3 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool3
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage4"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    relu4 = net
    print_inferred_shape(net, stage, bsize)

    pool4 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool4
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage5"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    net5 = net
    #pool5 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #net = pool5
    #print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Up-sampling net5 and merge with pool4
    stage = "up_stage6"
    #print("up_stage6:\nup-pooling pool5")
    net = mx.sym.Deconvolution(net5, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=8*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("Crop net")
    net = mx.sym.Crop(net, relu4, num_args =2)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage6 net with relu4")
    net = mx.sym.Concat(*[relu4, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    net6 = net
    #############################################################################################################
    # Up-sampling net6 and merge with pool3
    stage = "up_stage7"
    #print("up_stage7:\nup-pooling stage6 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=4*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage7 net with relu3")
    net = mx.sym.Concat(*[relu3, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    #############################################################################################################
    # Up-sampling net7 and merge with pool2
    stage = "up_stage8"
    #print("up_stage8:\nup-pooling stage7 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=2*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage8 net with relu2")
    net = mx.sym.Concat(*[relu2, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Up-sampling net8 and merge with pool1
    stage = "up_stage9"
    #print("up_stage9:\nup-pooling stage8 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    print_inferred_shape(relu1, stage+" relu1")
    #print("Crop relu1")
    crop1 = mx.sym.Crop(relu1, net, num_args =2)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage9 net with crop1")
    net = mx.sym.Concat(*[crop1, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Final Stage:
    stage = "Final Stage:"
    pad_size = (0, 0)
    kernel_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    net = mx.sym.Flatten(net)
    print_inferred_shape(net, stage, bsize)

    #net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)
    net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    print_inferred_shape(net, stage, bsize)

    return net

def get_unet_debug(bsize=1):
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    stride = (1, 1)
    filter_count = 64
    work_space = 1024
    act_type="relu"

    #############################################################################################################
    stage = "down_stage1"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (0, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu1 = net
    print_inferred_shape(net, stage, bsize)

    pad_size = (0, 0)
    pool1 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), pad=pad_size)
    net = pool1
    print_inferred_shape(net, stage, bsize)


    #############################################################################################################
    stage = "down_stage2"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu2 = net
    print_inferred_shape(net, stage, bsize)

    pool2 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool2
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage3"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu3 = net
    print_inferred_shape(net, stage, bsize)

    pool3 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool3
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage4"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu4 = net
    print_inferred_shape(net, stage, bsize)

    pool4 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool4
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    stage = "down_stage5"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    net5 = net
    #pool5 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #net = pool5
    #print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Up-sampling net5 and merge with pool4
    stage = "up_stage6"
    #print("up_stage6:\nup-pooling pool5")
    net = mx.sym.Deconvolution(net5, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=8*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("Crop net")
    net = mx.sym.Crop(net, relu4, num_args =2)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage6 net with relu4")
    net = mx.sym.Concat(*[relu4, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    net6 = net
    #############################################################################################################
    # Up-sampling net6 and merge with pool3
    stage = "up_stage7"
    #print("up_stage7:\nup-pooling stage6 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=4*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage7 net with relu3")
    net = mx.sym.Concat(*[relu3, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    #############################################################################################################
    # Up-sampling net7 and merge with pool2
    stage = "up_stage8"
    #print("up_stage8:\nup-pooling stage7 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=2*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #print("merge stage8 net with relu2")
    net = mx.sym.Concat(*[relu2, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Up-sampling net8 and merge with pool1
    stage = "up_stage9"
    #print("up_stage9:\nup-pooling stage8 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    print_inferred_shape(relu1, stage+" relu1")
    #print("Crop relu1")
    crop1 = mx.sym.Crop(relu1, net, num_args =2)
    print_inferred_shape(net, stage, bsize)
    print_inferred_shape(crop1, stage+" crop1")
    #print("merge stage9 net with crop1")
    net = mx.sym.Concat(*[crop1, net])
    print_inferred_shape(net, stage, bsize)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)

    pad_size = (2, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage, bsize)
    #############################################################################################################
    # Final Stage:
    stage = "Final Stage:"
    pad_size = (0, 0)
    kernel_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    net = mx.sym.Flatten(net)
    print_inferred_shape(net, stage, bsize)

    #net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)
    net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    print_inferred_shape(net, stage, bsize)

    return net

if __name__ == "__main__":
    get_unet_large(bsize = settings.SEGMENT_BATCH_SIZE)
