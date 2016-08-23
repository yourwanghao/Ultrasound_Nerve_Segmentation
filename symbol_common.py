import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
import cv2
import os


def init_logger(logfile):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=logfile,
                        filemode='a')
    logger = logging.getLogger()

def print_inferred_shape(net, stage="stage1", bsize=settings.SEGMENT_BATCH_SIZE):
    ar, ou, au = net.infer_shape(data=(bsize, 1, settings.SCALE_HEIGHT, settings.SCALE_WIDTH))
    print(stage, ou)


def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True,
                       down_pool=False, up_pool=False, act_type="relu", convolution=True):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count,
                                   workspace=work_space)
        net = mx.sym.BatchNorm(net)
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type)
        print_inferred_shape(net)

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count,
                                  workspace=work_space)
        net = conv
        print_inferred_shape(conv)

    if batch_norm:
        net = mx.sym.BatchNorm(net)

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type)

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        net = pool
        print_inferred_shape(net)

    return net

def check_label_shapes(labels, preds, shape=0):
    """Check to see if the two arrays are the same size."""

    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

