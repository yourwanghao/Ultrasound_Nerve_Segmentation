SCALE_WIDTH = 96 
SCALE_HEIGHT= 96
ORIG_WIDTH = 580
ORIG_HEIGHT = 420

CROP_WIDTH= 128
CROP_HEIGHT= 128

TRAIN_LIST_PREFIX = "./tr.lst"
VALID_LIST_PREFIX = "./va.lst"
TRAIN_LIST_PREFIX_PROCESSED = "./tr.lst.processed"
VALID_LIST_PREFIX_PROCESSED = "./va.lst.processed"
FINAL_VALID_LIST = "./finalva.lst"
FINAL_TEST_LIST = "./test.lst"
FINAL_VALID_LIST_PROCESSED = "./finalva.lst.processed"
FINAL_TEST_LIST_PROCESSED = "./test.lst.processed"

BASE_DIR = "./"
AUGMENT_TRAIN_IMAGE_DIR = "./augment_image_dir/train/"
AUGMENT_VALID_IMAGE_DIR = "./augment_image_dir/valid/"
ORIG_IMAGE_DIR = "../Data/train/"
MEAN_TRAIN_IMAGE = "./other_image_dir/mean_train_image.tif"
FOLD_NUM = 1

CLASSIFY_TRAIN_EPOCHS = 60
CLASSIFY_BATCH_SIZE = 10
CLASSIFY_MODEL_NAME = "./classify_model/classifier"

LEARNING_RATE = 0.1*1e-2
SEGMENT_TRAIN_EPOCHS = 2
SEGMENT_BATCH_SIZE = 200
SEGMENT_MODEL_DIR = "./segment_model/"
SEGMENT_MODEL_NAME = "./segment_model/segmenter"

SEGMENT_VALID_DIR = "./segment_valid_dir/"

PRED_MASK_H5_FILE = "./pred_mask.h5"
PRED_TEST_MASK_H5_FILE = "./pred_test_mask.h5"

PREPARED_VALID_IMAGE_DIR = "./prepared_valid_image_dir/"
PREDICTED_VALID_MASK_DIR = "./predicted_valid_mask_dir/"
COARSE_PREDICTED_VALID_MASK_DIR = "./predicted_valid_mask_dir/coarse/"
FINAL_PREDICTED_VALID_MASK_DIR = "./predicted_valid_mask_dir/final/"

PREPARED_TEST_IMAGE_DIR = "./prepared_test_image_dir/"
ORIG_TEST_IMAGE_DIR = "../Data/test/"
PREDICTED_TEST_MASK_DIR = "./predicted_test_mask_dir/"
COARSE_PREDICTED_TEST_MASK_DIR = "./predicted_test_mask_dir/coarse/"
FINAL_PREDICTED_TEST_MASK_DIR = "./predicted_test_mask_dir/final/"

FINAL_ESSEMBLE_TEST_MASK_DIR = "./essemble_test_mask_dir/coarse/"

LOG_PERIOD = 1




