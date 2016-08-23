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

def combine_fold_mask(file_list, threshold):
    if file_list == settings.FINAL_TEST_LIST:
        imgdir = settings.ORIG_TEST_IMAGE_DIR
        maskdir = settings.PREDICTED_TEST_MASK_DIR
        outputdir = settings.COARSE_PREDICTED_TEST_MASK_DIR
    elif file_list == settings.FINAL_VALID_LIST:
        imgdir = settings.ORIG_IMAGE_DIR
        maskdir = settings.PREDICTED_VALID_MASK_DIR
        outputdir = settings.COARSE_PREDICTED_VALID_MASK_DIR

    if (os.path.exists(outputdir)==False):
        os.makedirs(outputdir) 

    test_list = list(pandas.read_csv(file_list, delimiter=",", header=None)[0])
    fold_range=range(settings.FOLD_NUM)
    ratio = 0
    for f in test_list:
        desfile = os.path.join(outputdir, os.path.basename(f)).replace('.tif', '_mask.tif')
        files = [f.replace(imgdir, maskdir+"fold"+str(fid)+"/").replace('.tif', '_mask.tif') for fid in fold_range]
        img = np.zeros((settings.SCALE_HEIGHT, settings.SCALE_WIDTH), dtype=np.float)
        for i in range(len(files)): 
            print(files[i])
            tmpimg = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE).astype(np.float)/255
            img += tmpimg

        img /= len(files)
        img[img>0.5] = 1
        img[img<=0.5] = 0
        if (img.sum()>threshold):
            ratio += 1
        else:
            img.fill(0)
        img *= 255
        img = img.astype(np.uint8)
        print('saving %s'%desfile)
        cv2.imwrite(desfile, img)
    ratio /= len(test_list)

    print("complete, has_mask ratio is %f"%ratio)

def dice_coef(mask, pred):
    pred = pred.flatten().astype(np.int32)
    mask = mask.flatten().astype(np.int32)

    smooth = 10
    ret = (2 * (pred * mask).sum() + smooth)/ ((pred.sum() + mask.sum())+smooth)

    return ret

def compute_dcs(file_list):
    '''
    Compute dice coeffient for file_list

    Note: If file_list is ./finalva.lst.processed, then the combined predicted mask files 
    are in settings.COARSE_PREDICTED_VALID_MASK_DIR. If file_list is ./test.lst.processed, 
    then the combined mask files are in settings.COARSE_PREDICTED_TEST_MASK_DIR
    '''
    if file_list == settings.FINAL_TEST_LIST:
        predicted_dir = settings.COARSE_PREDICTED_TEST_MASK_DIR
    elif file_list == settings.FINAL_VALID_LIST:
        predicted_dir = settings.COARSE_PREDICTED_VALID_MASK_DIR

    orig_mask_list = list(pandas.read_csv(file_list, delimiter=",", header=None)[1])

    best_dc = -1
    best_th = -1
    for threshold in range(1000,6000,1000):
        all_dc = 0
        for f in orig_mask_list:
            mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255
            pred_file = f.replace(os.path.dirname(f), predicted_dir)
            pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.resize(pred_mask, (settings.ORIG_WIDTH, settings.ORIG_HEIGHT), cv2.INTER_CUBIC)/255
            pred_mask[pred_mask>0] = 1
            if (pred_mask.sum()<threshold):
                pred_mask.fill(0)
            dc = dice_coef(mask, pred_mask)
            all_dc += dc
        mean_dc = all_dc / len(orig_mask_list)
        print("threshold %d: final mean dc is %f"%(threshold, mean_dc))
        if mean_dc > best_dc:
            best_dc = mean_dc
            best_th = threshold
    return best_dc, best_th

if __name__ == "__main__":
    import sys
    import argparse

    init_logger('combine_evaluate.log')
    parser = argparse.ArgumentParser(description='Parses parameters')
    parser.add_argument('--file_list', type=str,default="./finalva.lst", help = "the image file list")
    parser.add_argument('--threshold', type=int,default=0, help = "the threshold to filter the masks")
    parser.add_argument('--compute_dcs', type=int, default=1, help = "whether to compute dcs for model evaluation")
    args = parser.parse_args()

    print("Invoked as %s --file_list %s"%(sys.argv[0], args.file_list))
    combine_fold_mask(args.file_list, args.threshold)

    if args.compute_dcs == 1:
        best_dc, best_th = compute_dcs(args.file_list)
        print("best dice coefficient is %f at threshold %d"%(best_dc, best_th))
        logging.debug("best dice coefficient is %f at threshold %d"%(best_dc, best_th))

    
#    for fid in range(settings.FOLD_NUM):
#        get_mask_ratio_by_folder(settings.PREDICTED_TEST_MASK_DIR+'fold'+str(fid))
#
#    get_mask_ratio_by_folder(settings.COARSE_PREDICTED_TEST_MASK_DIR)

