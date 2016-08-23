#!/usr/bin/env python

import os
import settings
import numpy as np
from sklearn.cross_validation import KFold
import random
import glob
import csv
import pandas
import cv2

def get_patient_list():
    '''
    Get patient list of original training images in settings.ORIG_IMAGE_DIR dir
    '''
    all_train_files = [os.path.basename(f) for f in os.listdir(settings.ORIG_IMAGE_DIR) if "mask" not in f and ".tif" in f]

    patient_list = [ int(f.split('_')[0]) for f in all_train_files]
    patient_list = np.unique(np.sort(patient_list))

    return patient_list

def split_patient():
    '''
    Split patient list to settings.FOLD_NUM folds
    '''
    pid = get_patient_list()
    rnd = random.Random()
    rnd.seed(1234)
    random.shuffle(pid)

    if (settings.FOLD_NUM == 1):
        train_pid = pid[0:int(len(pid)*0.80)]
        valid_pid = pid[int(len(pid)*0.80)+1:]
        result = []
        result.append((train_pid, valid_pid))
    else:
        kfold = KFold(len(pid), n_folds=settings.FOLD_NUM, shuffle=False, random_state=1234)
        result = []
        for train_index, valid_index in kfold:
            print("pid",type(pid), len(pid))
            print("train_index", type(train_index), len(train_index))
            result.append((pid[train_index], pid[valid_index]))

    return result

def get_train_files(train_list_name="./tr.lst", valid_list_name="./va.lst"):
    '''
    Split original training image files according to patient id. 

    The image files list are saved to tr.lst and va.lst
    '''
    folds = split_patient()
    all_train_files = [os.path.join(settings.ORIG_IMAGE_DIR,f) for f in os.listdir(settings.ORIG_IMAGE_DIR) if "mask" not in f and ".tif" in f]
    print("all_train_files",len(all_train_files))

    fid = -1
    for fold in folds:
        fid += 1
        tr_pid = fold[0]
        va_pid = fold[1]

        print(tr_pid, va_pid)

        train_files=[]
        for pid in tr_pid:
            train_files+=([(f,f.replace('.tif',"_mask.tif")) for f in all_train_files if "/"+str(pid)+"_" in f])
        train_files = train_files[0:]
        fo = csv.writer(open(train_list_name+str(fid), "w"), delimiter=',', lineterminator='\n')
        for item in train_files:
            fo.writerow(item)

        valid_files=[]
        for pid in va_pid:
            valid_files+=([[f,f.replace('.tif',"_mask.tif")] for f in all_train_files if "/"+str(pid)+"_" in f])
        valid_files = valid_files[0:]
        fo = csv.writer(open(valid_list_name+str(fid), "w"), delimiter=',', lineterminator='\n')
        for item in valid_files:
            fo.writerow(item)

    return

def get_mixed_train_files():
    '''
    Split original training image files randomly

    The image files list are saved to tr.lst and va.lst
    '''
    rnd = random.Random()
    rnd.seed(1234)

    all_train_files = [os.path.join(settings.ORIG_IMAGE_DIR,f) for f in os.listdir(settings.ORIG_IMAGE_DIR) if "mask" not in f and ".tif" in f]
    all_train_files = np.array(all_train_files)
    print("all_train_files",len(all_train_files))
    all_test_files = [os.path.join(settings.ORIG_TEST_IMAGE_DIR,f) for f in os.listdir(settings.ORIG_TEST_IMAGE_DIR) if "mask" not in f and ".tif" in f]
    all_test_files = np.array(all_test_files)
    print("all_test_files",len(all_test_files))

    #Split original train images to train and final valid images
    #Final valid images are used to evaluate the essembled model
    train_files = all_train_files[0:int(len(all_train_files)*0.80)]
    final_valid_files = all_train_files[(int(len(all_train_files)*0.80)+1):]
    final_valid_files = ([(f,f.replace('.tif',"_mask.tif")) for f in final_valid_files])
    final_test_files = ([(f,-1) for f in all_test_files])
    #Save the final valid list
    fo = csv.writer(open(settings.FINAL_VALID_LIST, "w"), delimiter=',', lineterminator='\n')
    for item in final_valid_files:
        fo.writerow(item)
    #Save the final valid list
    fo = csv.writer(open(settings.FINAL_TEST_LIST, "w"), delimiter=',', lineterminator='\n')
    for item in final_test_files:
        fo.writerow(item)

    all_train_files = train_files
    if (settings.FOLD_NUM == 1):
        train_f = all_train_files[0:int(len(all_train_files)*0.80)]
        valid_f= all_train_files[(int(len(all_train_files)*0.80)+1):len(all_train_files)]

        train_files=[]
        train_files+=([(f,f.replace('.tif',"_mask.tif")) for f in train_f])
        valid_files=[]
        valid_files+=([(f,f.replace('.tif',"_mask.tif")) for f in valid_f])

        fid = 0
        fo = csv.writer(open(settings.TRAIN_LIST_PREFIX+str(fid), "w"), delimiter=',', lineterminator='\n')
        for item in train_files:
            fo.writerow(item)

        fo = csv.writer(open(settings.VALID_LIST_PREFIX+str(fid), "w"), delimiter=',', lineterminator='\n')
        for item in valid_files:
            fo.writerow(item)
    else:
        folds= KFold(len(all_train_files), n_folds=settings.FOLD_NUM, shuffle=False, random_state=None)

        fid = -1
        for train_index, valid_index in folds:
            fid += 1
            print(fid)
            tf =all_train_files[train_index]
            vf =all_train_files[valid_index]
            train_files=[]
            train_files+=([(f,f.replace('.tif',"_mask.tif")) for f in tf])
            valid_files=[]
            valid_files+=([(f,f.replace('.tif',"_mask.tif")) for f in vf])
    
            fo = csv.writer(open(settings.TRAIN_LIST_PREFIX+str(fid), "w"), delimiter=',', lineterminator='\n')
            for item in train_files:
                fo.writerow(item)
    
            fo = csv.writer(open(settings.VALID_LIST_PREFIX+str(fid), "w"), delimiter=',', lineterminator='\n')
            for item in valid_files:
                fo.writerow(item)

    return

def get_mask_ratio(fid=0, list_prefix=settings.VALID_LIST_PREFIX):
    '''
    Get the ratio of "has mask" image in specific list

    The list file names are like tr.lst0, tr.lst1, etc.
    By changing the list prefix, it will be va.lst0, va.lst1, etc.
    '''
    valid_mask_list = list(pandas.read_csv(list_prefix+str(fid), delimiter=",", header=None)[1])
    dstdir_valid = settings.AUGMENT_VALID_IMAGE_DIR+"fold"+str(fid)+"/"
    orgdir_valid = settings.ORIG_IMAGE_DIR
    preddir_valid = settings.PREDICTED_VALID_MASK_DIR+"fold"+str(fid)+"/"
    orig_mask_list = [f.replace(dstdir_valid, orgdir_valid) for f in valid_mask_list]
    pred_mask_list = [f.replace(dstdir_valid, preddir_valid) for f in valid_mask_list]


    pred_has_mask = 0
    has_mask = 0
    mask_pixel = 0
    for f in orig_mask_list:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if (mask.sum()>0):
            has_mask += 1
            mask_pixel += mask.sum()/255

    has_mask_ratio = has_mask/ len(orig_mask_list)
    mask_pixel_ratio = mask_pixel /(len(orig_mask_list)*settings.ORIG_WIDTH*settings.ORIG_HEIGHT)
    print("has_mask ratio for %s%d is %f, mask_pixel ratio is %f"%(list_prefix, fid, has_mask_ratio, mask_pixel_ratio))

def get_mask_ratio_by_image_folder(image_folder):
    '''
    Get the ratio of "has mask" image in specific image folder
    '''
    mask_files = [os.path.join(image_folder,f) for f in os.listdir(image_folder) if
                   '_mask.tif' in f]
    has_mask = 0
    for f in mask_files:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if (mask.sum()>0):
            has_mask += 1

    has_mask_ratio = has_mask/ len(mask_files)
    print("has_mask ratio for image_folder %s is %f"%(image_folder, has_mask_ratio))

if __name__ == "__main__":
    # Experiment shows that randomly split works better 
    get_mixed_train_files()
#    get_train_files()
#    for fid in range(settings.FOLD_NUM):
#        get_mask_ratio(fid, list_prefix="tr.lst")
#        get_mask_ratio(fid, list_prefix="va.lst")
    #get_mask_ratio_by_folder('./predicted_test_mask_dir/fold2')

