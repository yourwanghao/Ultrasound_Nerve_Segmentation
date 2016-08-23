#!/usr/bin/env python

import numpy as np
import cv2
import math
import pandas
import settings
import os
import random
import csv
import shutil
from symbol_common import *

def process_image(srcfile, save_dir=None, isTrain=True):
    '''
    Process images and masks for better learning result

    Currently only resize image and mask, and do clahe for resized images
    '''
    print(srcfile)

    image = cv2.imread(srcfile, cv2.IMREAD_UNCHANGED)
    desfile= os.path.join(save_dir, os.path.basename(srcfile))
    rows, cols = image.shape
    if (cols != settings.SCALE_WIDTH) or (rows != settings.SCALE_HEIGHT):
        image = cv2.resize(image, (settings.SCALE_WIDTH, settings.SCALE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(tileGridSize=(1, 1))
    image = clahe.apply(image)
    cv2.imwrite(desfile, image)

    has_mask = 0
    if isTrain==True:
        mask_f = srcfile.replace('.tif', '_mask.tif')
        mask_image = cv2.imread(mask_f, cv2.IMREAD_UNCHANGED)
        has_mask = 0 
        if (mask_image.sum()>0):
            has_mask = 1
        mask_f = os.path.join(save_dir, os.path.basename(mask_f))
        if (cols != settings.SCALE_WIDTH) or (rows != settings.SCALE_HEIGHT):
            mask_image = cv2.resize(mask_image, (settings.SCALE_WIDTH, settings.SCALE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(mask_f, mask_image)

    return has_mask

def preprocess_final_valid_images():
    '''
    Preprocess images and masks of final validate images

    We save the processed images/masks in directories prefix as settings.PREPARED_VALID_IMAGE_DIR

    And we save the file list in settings.FINAL_VALID_LIST_PROCESSED 
    '''
    dstdir_final_valid = settings.PREPARED_VALID_IMAGE_DIR
    if (os.path.exists(dstdir_final_valid)):
        shutil.rmtree(dstdir_final_valid)
    os.makedirs(dstdir_final_valid)

    final_valid_files= pandas.read_csv(settings.FINAL_VALID_LIST, header=None)[0]
    file_with_mask_no = 0
    for f in final_valid_files:
        print("for %s, handle %s"%(settings.FINAL_VALID_LIST,f))
        has_mask=process_image(f, save_dir=dstdir_final_valid)
        file_with_mask_no += has_mask
     
    logging.debug("In %s: with mask file number is %d, ratio is %f"%(settings.FINAL_VALID_LIST, file_with_mask_no, file_with_mask_no/len(final_valid_files)))
    print("In %s: with mask file number is %d, ratio is %f"%(settings.FINAL_VALID_LIST, file_with_mask_no, file_with_mask_no/len(final_valid_files)))
    
    final_valid_files = [f.replace(settings.ORIG_IMAGE_DIR, dstdir_final_valid) for f in final_valid_files]
    final_valid_data = [(f,f.replace('.tif','_mask.tif')) for f in final_valid_files]
    final_valid_list_file = settings.FINAL_VALID_LIST_PROCESSED
    fo = csv.writer(open(final_valid_list_file, "w"), delimiter=',', lineterminator='\n')
    for item in final_valid_data:
        fo.writerow(item)

def preprocess_test_images():
    '''
    Preprocess images and masks of test images

    We save the processed images/masks in directories prefix as settings.PREPARED_TEST_IMAGE_DIR

    And we save the file list in settings.FINAL_TEST_LIST_PROCESSED 
    '''
    dstdir_test = settings.PREPARED_TEST_IMAGE_DIR
    if (os.path.exists(dstdir_test)):
        shutil.rmtree(dstdir_test)
    os.makedirs(dstdir_test)

    test_files= pandas.read_csv(settings.FINAL_TEST_LIST, header=None)[0]
    file_with_mask_no = 0
    for f in test_files:
        print("for %s, handle %s"%(settings.FINAL_TEST_LIST,f))
        process_image(f, save_dir=dstdir_test, isTrain=False)

    test_files = [f.replace(settings.ORIG_TEST_IMAGE_DIR, dstdir_test) for f in test_files]
    test_data = [(f,-1) for f in test_files]
    test_list_file = settings.FINAL_TEST_LIST_PROCESSED
    fo = csv.writer(open(test_list_file, "w"), delimiter=',', lineterminator='\n')
    for item in test_data:
        fo.writerow(item)

def preprocess_train_images():
    '''
    Preprocess images and masks in each training fold

    We saved the processed images/masks in directories prefix as settings.AUGMENT_TRAIN_IMAGE_DIR
    And modifying the va.lst and tr.lst accordingly
    '''
    for fid in range(settings.FOLD_NUM):
        # Clean and create output directories
        dstdir_train = settings.AUGMENT_TRAIN_IMAGE_DIR+"fold"+str(fid)+"/"
        if (os.path.exists(dstdir_train)):
            shutil.rmtree(dstdir_train)
        os.makedirs(dstdir_train)
        dstdir_valid = settings.AUGMENT_VALID_IMAGE_DIR+"fold"+str(fid)+"/"
        if (os.path.exists(dstdir_valid)):
            shutil.rmtree(dstdir_valid)
        os.makedirs(dstdir_valid)

        # Process training images
        train_list_file = settings.TRAIN_LIST_PREFIX+str(fid)
        valid_list_file = settings.VALID_LIST_PREFIX+str(fid)
        train_data = pandas.read_csv(train_list_file, header=None)
        train_files = list(train_data[0])
        file_with_mask_no = 0
        for f in train_files:
            print("for %s, handle %s"%(train_list_file,f))
            has_mask=process_image(f, save_dir=dstdir_train)
            file_with_mask_no += has_mask

        logging.debug("In %s: with mask file number is %d, ratio is %f"%(train_list_file, file_with_mask_no, file_with_mask_no/len(train_files)))
        print("In %s: with mask file number is %d, ratio is %f"%(train_list_file, file_with_mask_no, file_with_mask_no/len(train_files)))

        # Process validation images
        valid_data = pandas.read_csv(valid_list_file, header=None)
        valid_files = list(valid_data[0])
        file_with_mask_no = 0
        for f in valid_files:
            print("for %s, handle %s"%(valid_list_file,f))
            has_mask=process_image(f, save_dir=dstdir_valid)
            file_with_mask_no += has_mask
        logging.debug("In %s: with mask file number is %d, ratio is %f"%(valid_list_file, file_with_mask_no, file_with_mask_no/len(valid_files)))
        print("In %s: with mask file number is %d, ratio is %f"%(valid_list_file, file_with_mask_no, file_with_mask_no/len(valid_files)))

        #Modify original tr.lst and va.lst
        train_files = [f.replace(settings.ORIG_IMAGE_DIR, dstdir_train) for f in train_files]
        train_data = [(f,f.replace('.tif','_mask.tif')) for f in train_files]

        valid_files = [f.replace(settings.ORIG_IMAGE_DIR, dstdir_valid) for f in valid_files]
        valid_data = [(f,f.replace('.tif','_mask.tif')) for f in valid_files]

        train_list_file = settings.TRAIN_LIST_PREFIX_PROCESSED+str(fid)
        valid_list_file = settings.VALID_LIST_PREFIX_PROCESSED+str(fid)
        fo = csv.writer(open(train_list_file, "w"), delimiter=',', lineterminator='\n')
        for item in train_data:
            fo.writerow(item)
        fo = csv.writer(open(valid_list_file, "w"), delimiter=',', lineterminator='\n')
        for item in valid_data:
            fo.writerow(item)

if __name__ == "__main__":
    init_logger('step1.log')
    preprocess_final_valid_images()
    preprocess_test_images()
    preprocess_train_images()
