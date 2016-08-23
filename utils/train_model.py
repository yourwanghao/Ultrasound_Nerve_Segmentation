#!/usr/bin/env python

import settings
import numpy as np
import mxnet as mx
import logging
from fileiter import FileIter
import cv2
import os
import pandas
import shutil

import sys
sys.path.append(".")
from symbols.symbol_unet_small import get_unet_small
from symbols.symbol_common import *

import gc
import sys

mx.random.seed(1301)

class DCCustom(mx.metric.EvalMetric):
    """
    Custom class to caculate dice coeffient
    """

    def __init__(self):
        super(DCCustom, self).__init__('dicecoef')
        self.epoch = 0

    def dice_coef(self, label, pred):
        pred = pred.flatten().astype(np.int32)
        label = label.flatten().astype(np.int32)

        smooth = 10
        ret = (2 * (pred * label).sum() + smooth)/ ((pred.sum() + label.sum())+smooth)

        return ret

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            label = labels[i].asnumpy().astype('int32')
            pred = preds[i].asnumpy()
            pred_label = np.argmax(pred, axis=1)

            check_label_shapes(label, pred_label)

            dc = 0
            for file_id in range(len(label)):
                dc += self.dice_coef(label[file_id], pred_label[file_id])
            self.sum_metric += dc
            self.num_inst += len(label)
            self.epoch += 1
            print("In DCCustom epoch = %d, dc=%f, curr dc=%f" % (self.epoch, self.sum_metric/self.num_inst, dc/len(label)))

def continue_train(target_epoch = 5, begin_epoch = 1):
    '''
    Continue train segment model from begin_epoch

    This function is used in case if we want to restart a stopped training process 
    '''
    network = get_unet_small()
    devs = [mx.gpu(0)]
    epochs = target_epoch

    for fid in range(settings.FOLD_NUM):
        eval_metric = DCCustom(),
        data_train = FileIter(root_dir=settings.BASE_DIR, flist_name="./tr.lst"+str(fid),
                              batch_size=settings.SEGMENT_BATCH_SIZE,
                              augment = True)

        data_valid = FileIter(root_dir=settings.BASE_DIR, flist_name="./va.lst"+str(fid),
                              batch_size=settings.SEGMENT_BATCH_SIZE,
                              augment = False)

        train_list = list(pandas.read_csv(data_train.get_flist_name(), delimiter=",", header=None)[0])
        e_cb = [mx.callback.do_checkpoint(settings.SEGMENT_MODEL_NAME+"fold"+str(fid))]
        b_cb = [mx.callback.log_train_metric(round(len(train_list)/settings.SEGMENT_BATCH_SIZE))]
        batch_per_epoch = round(len(train_list)/settings.SEGMENT_BATCH_SIZE)

        symbol, arg_params, aux_params = mx.model.load_checkpoint(settings.SEGMENT_MODEL_NAME+"fold"+str(fid), begin_epoch)
        train_model = mx.model.FeedForward(
            ctx=devs,
            symbol=symbol,
            arg_params=arg_params,
            aux_params=aux_params,
            num_epoch=epochs,
            begin_epoch=begin_epoch,
            optimizer="adam",
            initializer=mx.initializer.Normal(sigma = np.sqrt(2/(3*3*32))),
            learning_rate= settings.LEARNING_RATE*(0.999**(batch_per_epoch*begin_epoch-1)),
            wd=0,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(step=1, factor=0.999)
        )

        import time
        start_time = time.time()
        train_model.fit(
            X=data_train,
            eval_data=data_valid,
            eval_metric=DCCustom(),
            batch_end_callback=b_cb,
            epoch_end_callback=e_cb
        )
        end_time = time.time()

        print("Fold: %d, Number of Epochs: %d, Training time: %f seconds" % (fid, settings.SEGMENT_TRAIN_EPOCHS, end_time - start_time))

        print("done")
        train_model = None
        gc.collect()

def train_segment_model(fid):
    '''
    Train segment model 
    '''
    if (os.path.exists(settings.SEGMENT_MODEL_DIR)):
        shutil.rmtree(settings.SEGMENT_MODEL_DIR)
    os.makedirs(settings.SEGMENT_MODEL_DIR)

    network = get_unet_small()
    devs = [mx.gpu(0)]

    print("Begin train for fold %d"%fid)
    data_train = FileIter(root_dir=settings.BASE_DIR, flist_name=settings.TRAIN_LIST_PREFIX_PROCESSED+str(fid),
                          batch_size=settings.SEGMENT_BATCH_SIZE,
                          augment = True, random_crop = False)

    data_valid = FileIter(root_dir=settings.BASE_DIR, flist_name=settings.VALID_LIST_PREFIX_PROCESSED+str(fid),
                          batch_size=settings.SEGMENT_BATCH_SIZE,
                          augment = False, random_crop = False)

    train_list = list(pandas.read_csv(data_train.get_flist_name(), delimiter=",", header=None)[0])
    e_cb = [mx.callback.do_checkpoint(settings.SEGMENT_MODEL_NAME+"fold"+str(fid))]
    b_cb = [mx.callback.log_train_metric(int(len(train_list)/settings.SEGMENT_BATCH_SIZE))]

    train_model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=settings.SEGMENT_TRAIN_EPOCHS,
        optimizer="adam",
        initializer=mx.initializer.Normal(sigma = np.sqrt(2/(3*3*32))),
        learning_rate= settings.LEARNING_RATE,
        wd=0,
        lr_scheduler=mx.lr_scheduler.FactorScheduler(step=1, factor=0.999)
    )

    import time
    start_time = time.time()
    train_model.fit(
        X=data_train,
        eval_data=data_valid,
        eval_metric=DCCustom(),
        batch_end_callback=b_cb,
        epoch_end_callback=e_cb
    )
    end_time = time.time()

    print("Fold: %d, Number of Epochs: %d, Training time: %f seconds" % (fid, settings.SEGMENT_TRAIN_EPOCHS, end_time - start_time))

    print("done")
    train_model = None
    gc.collect()

if __name__ == "__main__":
    init_logger('step2.log')
    fid = 0
    if (len(sys.argv)>1):
        fid = int(sys.argv[1])
    print("train for ", fid)
    train_segment_model(fid)
    #continue_train(70, begin_epoch = 40)
