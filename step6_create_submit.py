#!/usr/bin/env python
import os
import utils.settings as settings
import numpy as np
from sklearn.cross_validation import KFold
import random
import glob
import csv
import cv2
import pickle
import pandas

import mxnet as mx
import h5py
import argparse



# Any results you write to the current directory are saved as output.
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

import datetime
def create_submit(path=settings.FINAL_PREDICTED_TEST_MASK_DIR):
    test_mask_files = [os.path.join(path, f) for f in os.listdir(path) if
                       '_mask.tif' in f]

    sub_file = os.path.join(
        'submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    subm.write("img,pixels\n")

    for f in test_mask_files:
        print(f)
        test_id = os.path.basename(f)[:-4].replace("_mask", "")
        subm.write(str(test_id) + ',')
        mask = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if mask.sum() != 0:
            encode = RLenc(mask)
            subm.write(encode)
        subm.write('\n')
    subm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='final image path')
    parser.add_argument('--image-path', type=str,default=settings.FINAL_PREDICTED_TEST_MASK_DIR,
                        help='the path of final image')
    args = parser.parse_args()
    
    path = args.image_path
    create_submit(path)
