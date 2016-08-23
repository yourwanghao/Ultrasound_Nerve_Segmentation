import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
from classify_fileiter import CFileIter
import cv2
import os

from symbol_common import *

def get_unet_orig():
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
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu1 = net
    print_inferred_shape(net, stage)

    pad_size = (0, 0)
    pool1 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), pad=pad_size)
    net = pool1
    print_inferred_shape(net, stage)


    #############################################################################################################
    stage = "down_stage2"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu2 = net
    print_inferred_shape(net, stage)

    pool2 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool2
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage3"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (0, 0)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu3 = net
    print_inferred_shape(net, stage)

    pool3 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool3
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage4"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu4 = net
    print_inferred_shape(net, stage)

    pool4 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool4
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage5"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    net5 = net
    #pool5 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #net = pool5
    #print_inferred_shape(net, stage)
    #############################################################################################################
    # Up-sampling net5 and merge with pool4
    stage = "up_stage6"
    #print("up_stage6:\nup-pooling pool5")
    net = mx.sym.Deconvolution(net5, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=8*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("Crop net")
    net = mx.sym.Crop(net, relu4, num_args =2)
    print_inferred_shape(net, stage)
    #print("merge stage6 net with relu4")
    net = mx.sym.Concat(*[relu4, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    net6 = net
    #############################################################################################################
    # Up-sampling net6 and merge with pool3
    stage = "up_stage7"
    #print("up_stage7:\nup-pooling stage6 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=4*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("merge stage7 net with relu3")
    net = mx.sym.Concat(*[relu3, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (2, 2)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    #############################################################################################################
    # Up-sampling net7 and merge with pool2
    stage = "up_stage8"
    #print("up_stage8:\nup-pooling stage7 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=2*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("merge stage8 net with relu2")
    net = mx.sym.Concat(*[relu2, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #############################################################################################################
    # Up-sampling net8 and merge with pool1
    stage = "up_stage9"
    #print("up_stage9:\nup-pooling stage8 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    print_inferred_shape(relu1, stage+" relu1")
    #print("Crop relu1")
    crop1 = mx.sym.Crop(relu1, net, num_args =2)
    print_inferred_shape(net, stage)
    #print("merge stage9 net with crop1")
    net = mx.sym.Concat(*[crop1, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
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
    print_inferred_shape(net, stage)

    #net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)
    net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    print_inferred_shape(net, stage)

    return net

def get_unet_debug():
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
    print_inferred_shape(net, stage)

    pad_size = (0, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu1 = net
    print_inferred_shape(net, stage)

    pad_size = (0, 0)
    pool1 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), pad=pad_size)
    net = pool1
    print_inferred_shape(net, stage)


    #############################################################################################################
    stage = "down_stage2"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu2 = net
    print_inferred_shape(net, stage)

    pool2 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool2
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage3"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu3 = net
    print_inferred_shape(net, stage)

    pool3 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool3
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage4"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    relu4 = net
    print_inferred_shape(net, stage)

    pool4 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    net = pool4
    print_inferred_shape(net, stage)
    #############################################################################################################
    stage = "down_stage5"
    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=16*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    net5 = net
    #pool5 = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #net = pool5
    #print_inferred_shape(net, stage)
    #############################################################################################################
    # Up-sampling net5 and merge with pool4
    stage = "up_stage6"
    #print("up_stage6:\nup-pooling pool5")
    net = mx.sym.Deconvolution(net5, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=8*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("Crop net")
    net = mx.sym.Crop(net, relu4, num_args =2)
    print_inferred_shape(net, stage)
    #print("merge stage6 net with relu4")
    net = mx.sym.Concat(*[relu4, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=8*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    net6 = net
    #############################################################################################################
    # Up-sampling net6 and merge with pool3
    stage = "up_stage7"
    #print("up_stage7:\nup-pooling stage6 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=4*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("merge stage7 net with relu3")
    net = mx.sym.Concat(*[relu3, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=4*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    #############################################################################################################
    # Up-sampling net7 and merge with pool2
    stage = "up_stage8"
    #print("up_stage8:\nup-pooling stage7 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=2*filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #print("merge stage8 net with relu2")
    net = mx.sym.Concat(*[relu2, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=2*filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    #############################################################################################################
    # Up-sampling net8 and merge with pool1
    stage = "up_stage9"
    #print("up_stage9:\nup-pooling stage8 result")
    net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                               workspace=work_space)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
    print_inferred_shape(relu1, stage+" relu1")
    #print("Crop relu1")
    crop1 = mx.sym.Crop(relu1, net, num_args =2)
    print_inferred_shape(net, stage)
    print_inferred_shape(crop1, stage+" crop1")
    #print("merge stage9 net with crop1")
    net = mx.sym.Concat(*[crop1, net])
    print_inferred_shape(net, stage)

    pad_size = (1, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)

    pad_size = (2, 1)
    conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                              workspace=work_space)
    net = conv
    net = mx.sym.Activation(net, act_type=act_type)
    print_inferred_shape(net, stage)
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
    print_inferred_shape(net, stage)

    #net = mx.symbol.SoftmaxOutput(data=net, name='softmax', multi_output=True)
    net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
    print_inferred_shape(net, stage)

    return net

