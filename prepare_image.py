#!/usr/bin/env python
import os
import settings
import numpy as np
from sklearn.cross_validation import KFold
import random
import glob
import csv
import cv2

from symbol_unet_orig import get_unet_orig
from symbol_unet_large import get_unet_large
from symbol_unet_small import get_unet_small
from symbol_common import *

import gc

import pandas
from PIL import Image
import shutil


def predict_fold_mask(fid, min_epoch, threshold, input_list_name):
    epoch = min_epoch
    if input_list_name == settings.FINAL_TEST_LIST_PROCESSED:
        desdir = settings.PREDICTED_TEST_MASK_DIR
    elif input_list_name == settings.FINAL_VALID_LIST_PROCESSED:
        desdir = settings.PREDICTED_VALID_MASK_DIR

    desdir = desdir+"fold"+str(fid)
    if (os.path.exists(desdir)):
        shutil.rmtree(desdir)
    os.makedirs(desdir)

    bsize = 1
    net = get_unet_small(bsize)
    mod = mx.mod.Module(net, context=[mx.gpu(0)])
    print("fold %d, model %s, epoch %d"%(fid, settings.SEGMENT_MODEL_NAME+"fold"+str(fid), epoch))

    data_test = FileIter(root_dir=settings.BASE_DIR, flist_name=input_list_name,
                          batch_size=bsize,
                          augment=False, shuffle=False)

    sym, arg_params, aux_params = mx.model.load_checkpoint(settings.SEGMENT_MODEL_NAME+"fold"+str(fid), epoch)
    assert (arg_params != None) and (aux_params != None)
    mod.bind(data_shapes=data_test.provide_data, label_shapes=data_test.provide_label)
    mod.set_params(arg_params, aux_params, allow_missing=False)


    test_list = list(pandas.read_csv(data_test.get_flist_name(), delimiter=",", header=None)[0])
    file_id = 0
    for preds, i_batch, batch in mod.iter_predict(data_test):
        pred_label = preds[0].asnumpy()
        pred_label = np.argmax(pred_label, axis=1).astype(np.uint8)*255
        print("dtype", pred_label.dtype, "shape", pred_label.shape)

        for bid in range(bsize):
            desfile = os.path.join(desdir, os.path.basename(test_list[file_id].replace('.tif', '_mask.tif')))
            print("saving predicted mask for %s"%desfile)
            tmp = pred_label[bid].reshape((settings.SCALE_HEIGHT, settings.SCALE_WIDTH))
            if (tmp.sum()/255<threshold):
                tmp.fill(0)

            cv2.imwrite(desfile, tmp)
            file_id += 1

def combine_fold_mask():
    test_list = list(pandas.read_csv('./test.lst', delimiter=",", header=None)[0])
    #weights = list(pandas.read_csv('./weights.txt', header=None)[0])
    #weights = [1 for i in weights]
    #print(len(weights), weights)


    fold_range=range(settings.FOLD_NUM)
    #fold_range=[0,4,7,12,15,16,17]
    if (os.path.exists(settings.COARSE_PREDICTED_TEST_MASK_DIR)==False):
        os.makedirs(settings.COARSE_PREDICTED_TEST_MASK_DIR) 
    ratio = 0
    for f in test_list:
        desfile = os.path.join(settings.COARSE_PREDICTED_TEST_MASK_DIR, os.path.basename(f)).replace('.tif', '_mask.tif')
        #files = [f.replace(settings.PREPARED_TEST_IMAGE_DIR, settings.PREDICTED_TEST_MASK_DIR+"fold"+str(fid)+"/").replace('.tif', '_mask.tif') for fid in range(settings.FOLD_NUM)]
        files = [f.replace(settings.PREPARED_TEST_IMAGE_DIR, settings.PREDICTED_TEST_MASK_DIR+"fold"+str(fid)+"/").replace('.tif', '_mask.tif') for fid in fold_range]
        img = np.zeros((settings.SCALE_HEIGHT, settings.SCALE_WIDTH), dtype=np.float)
        for i in range(len(files)): 
            tmpimg = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE).astype(np.float)/255
            #img += weights[i]*tmpimg
            img += tmpimg

        #img /= sum(weights)
        img /= len(files)
        img[img>0.5] = 1
        img[img<=0.5] = 0
        if (img.sum()>0):
            ratio += 1
        img *= 255
        img = img.astype(np.uint8)
        print('saving %s'%desfile)
        cv2.imwrite(desfile, img)
    ratio /= len(test_list)

    print("complete, has_mask ratio is %f"%ratio)

def get_mask_ratio_by_folder(fold):
    mask_files = [os.path.join(fold,f) for f in os.listdir(fold) if
                   '_mask.tif' in f]
    has_mask = 0
    for f in mask_files:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if (mask.sum()>0):
            has_mask += 1

    has_mask_ratio = has_mask/ len(mask_files)
    print("has_mask ratio for fold %s is %f"%(fold, has_mask_ratio))

if __name__ == "__main__":
    #get_test_files()
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Parses parameters')
    parser.add_argument('--fold_id', type=int,default=0, help = "the fold id")
    parser.add_argument('--epoch', type=int,default=1, help = "the best score epoch")
    parser.add_argument('--threshold', type=int,default=2000, help = "the threshold of incorrect masks")
    parser.add_argument('--file_list', type=str,default="./finalva.lst.processed", help = "the image file list")
    args = parser.parse_args()

    print("Invoked as %s --fold_id %d --epoch %d --threshold %d --file_list %s "%
        (sys.argv[0], args.fold_id, args.epoch, args.threshold, args.file_list))

    predict_fold_mask(args.fold_id, args.epoch, args.threshold, args.file_list)
#    for fid in range(settings.FOLD_NUM):
#        get_mask_ratio_by_folder(settings.PREDICTED_TEST_MASK_DIR+'fold'+str(fid))
#
#    get_mask_ratio_by_folder(settings.COARSE_PREDICTED_TEST_MASK_DIR)

