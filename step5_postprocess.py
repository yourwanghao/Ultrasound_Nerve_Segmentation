#!/usr/bin/env python

import cv2
import utils.settings as settings
import numpy as np
import matplotlib.pyplot as plt
import os

def erode_image(img, kernel):
   eroded = cv2.erode(img,kernel)  
   return eroded

def dilate_image(img, kernel):
   dilated = cv2.dilate(img,kernel)  
   return dilated

def open_image(img, kernel):
   opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
   return opened

def close_image(img, kernel):
   closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
   return closed

def remove_outlier(img):
   ret, bwImg = cv2.threshold(img, 127, 255.0, cv2.THRESH_BINARY)
   im2, contours, hierarchy = cv2.findContours(bwImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   maxC = None
   maxArea = -1
   for c in contours:
       c = np.array(c)
       area = cv2.contourArea(c)

       print(area)
       if (area > maxArea):
           maxC = c
           maxArea = area

   im2.fill(0)
   cv2.drawContours(im2, [maxC], 0, (255,255,255), -1)
   return im2
#   plt.gray()
#   plt.imshow(im2)
#   plt.show()

def combine_rois():
  test_mask_files = [os.path.join(settings.COARSE_PREDICTED_TEST_MASK_DIR, f) for f in os.listdir(settings.COARSE_PREDICTED_TEST_MASK_DIR) if
                   '_mask.tif' in f] 

def postprocess(threshold):
  test_mask_files = [os.path.join(settings.COARSE_PREDICTED_TEST_MASK_DIR, f) for f in os.listdir(settings.COARSE_PREDICTED_TEST_MASK_DIR) if
                   '_mask.tif' in f] 
  if (os.path.exists(settings.FINAL_PREDICTED_TEST_MASK_DIR)==False):
        os.makedirs(settings.FINAL_PREDICTED_TEST_MASK_DIR)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
  perc = 0
  for f in test_mask_files:
      print(f)
      img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
      closed = close_image(img, kernel)
      if (closed.sum()/255.<threshold):
          print("%s's sum is smaller than %d"%(f, threshold))
          closed.fill(0)
      closed = remove_outlier(closed)

      if (closed.sum()>0):
          perc += 1
      des_file = f.replace(settings.COARSE_PREDICTED_TEST_MASK_DIR, settings.FINAL_PREDICTED_TEST_MASK_DIR)
      cv2.imwrite(des_file, closed)

  perc /= len(test_mask_files)
  print("has mask perc is %f"%perc)
      

if __name__ == "__main__":
    from step0_split_images import get_mask_ratio_by_image_folder
    postprocess(threshold = 0)
    get_mask_ratio_by_image_folder(settings.FINAL_PREDICTED_TEST_MASK_DIR)

