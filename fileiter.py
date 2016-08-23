import mxnet as mx
import numpy as np
from mxnet.io import DataIter
from mxnet.io import DataBatch
import random
import cv2
import settings
import os
import logging
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

np.random.seed(1301)

def test_FileIter():
    fid = 0
    data_train = FileIter(root_dir=settings.BASE_DIR, flist_name="./tr.lst"+str(fid),
                          batch_size=1,
                          augment = True, random_crop = True)
    return data_train


class FileIter(DataIter):
    def __init__(self, root_dir, flist_name,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=1,
                 shuffle=True,
                 augment=False,
                 random_crop = False):
        self.file_lines = []
        self.epoch = 0
        self.shuffle = shuffle
        self.label_files = []
        self.image_files = []
        super(FileIter, self).__init__()
        self.batch_size = batch_size
        self.random = random.Random()
        self.random.seed(1301)
        self.root_dir = root_dir
        self.flist_name = flist_name
        self.data_name = data_name
        self.label_name = label_name
        self.augment = augment
        self.random_crop = random_crop

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.cursor = -1
        self.read_lines()
        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            line = self.get_line()
            data_img_name, label_img_name = line.strip('\n').split(",")
            #logging.debug("Load image (%s, %s)"%(data_img_name, label_img_name))
            print("Load image (%s, %s)"%(data_img_name, label_img_name))
            d, l = self._read_img(data_img_name, label_img_name)
            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l

        res = list(data.items()), list(label.items())
        return res

#    ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.
#    def elastic_transform(self, image, alpha, sigma, random_state=None):
#        global ELASTIC_INDICES
#        shape = image.shape
#
#        if self.ELASTIC_INDICES == None:
#            if random_state is None:
#                random_state = np.random.RandomState(1301)
#
#            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
#            x = x.transpose()
#            y = y.transpose()
#            ELASTIC_INDICES = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
#        return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)

    # Function to distort image
    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
    
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
    
        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
        # include 4 standard deviations in the kernel (the default for ndimage.gaussian_filter)
        # OpenCV also requires an odd size for the kernel hence the "| 1" part
        blur_size = int(4*sigma) | 1
        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
        #dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        #dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)
    
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    
        image = image.astype(np.float32)
        dx = dx.astype(np.float32)
        dy = dy.astype(np.float32)
        #print("hawk debug", image.dtype, dx.dtype, dx.shape, dy.dtype, dx.shape, shape)
        #result = cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR).reshape(shape)
        result = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        return result

    def augment_image(self, img, label):
        if (label.sum()==0):
            return img, label

        rnd_val = self.random.randint(0, 100)

#        if (rnd_val > 50):
#            # Merge images into separete channels (shape will be (cols, rols, 2))
#            im_merge = np.concatenate((img[...,None], label[...,None]), axis=2)
#            im_merge_t = self.elastic_transform(im_merge, im_merge.shape[1]*2, im_merge.shape[1]*0.08, im_merge.shape[1]*0.08)
#            # Split image and mask
#            img = im_merge[...,0]
#            label = im_merge[...,1]
#            print("elastic transform")

        rnd_val = self.random.randint(0, 100)
        if (rnd_val > 50):
            img = np.fliplr(img)
            label = np.fliplr(label)
            print("fliplr")

        rnd_val = self.random.randint(0, 100)
        if (rnd_val > 50):
            img = np.flipud(img)
            label = np.flipud(label)
            print("flipud")

        return img, label

    def _read_img(self, img_name, label_name):
        img_path = os.path.join(self.root_dir, img_name)
        print("hawk", img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)  # Image.open(img_path).convert("L")
        if (label_name == "-1"):
            label_path = label_name
            label = np.zeros(img.shape)
        else:
            label_path = os.path.join(self.root_dir, label_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(float)  # Image.open(label_path).convert("L")

        if label.shape != img.shape:
            label = np.zeros(img.shape)

        top = ((int(settings.ORIG_HEIGHT / settings.CROP_HEIGHT)+1)*settings.CROP_HEIGHT - settings.ORIG_HEIGHT)>>1
        bottom = ((int(settings.ORIG_HEIGHT / settings.CROP_HEIGHT)+1)*settings.CROP_HEIGHT - settings.ORIG_HEIGHT)>>1
        left = ((int(settings.ORIG_WIDTH / settings.CROP_WIDTH)+1)*settings.CROP_WIDTH - settings.ORIG_WIDTH)>>1
        right= ((int(settings.ORIG_WIDTH / settings.CROP_WIDTH)+1)*settings.CROP_WIDTH - settings.ORIG_WIDTH)>>1


        if (self.random_crop==True):
            img = cv2.copyMakeBorder(img,top,bottom,left,right, cv2.BORDER_REFLECT)
            label = cv2.copyMakeBorder(label,top,bottom,left,right, cv2.BORDER_REFLECT)
            if (settings.SCALE_WIDTH>=settings.CROP_WIDTH):
                crop_x = self.random.randint(0, settings.SCALE_WIDTH-settings.CROP_WIDTH)
            else:
                crop_x = 0

            if (settings.SCALE_HEIGHT>=settings.CROP_HEIGHT):
                crop_y = self.random.randint(0, settings.SCALE_HEIGHT-settings.CROP_HEIGHT)
            else:
                crop_y = 0

            crop_width = settings.CROP_WIDTH
            crop_height = settings.CROP_HEIGHT

            img = img[crop_y:(crop_y+crop_height), crop_x:(crop_x+crop_width)]
            label = label[crop_y:(crop_y+crop_height), crop_x:(crop_x+crop_width)]
            print(crop_x, crop_width, crop_y, crop_height, img.shape, label.shape)
            
        if self.augment:
            img, label = self.augment_image(img, label)

        self.image_files.append(img_path)
        #img = img - self.mean
        #img = (img - img.min())/(img.max() - img.min())
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        img = np.expand_dims(img, axis=0)  # (1, c, h, w) or (1, h, w)

        self.label_files.append(label_path)

        label /= 255.
        label = np.array(label)  # (h, w)
        label = label.reshape(1, label.shape[0] * label.shape[1])


        return img, label

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        print("pdata", res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        print("plabel", res)
        return res

    def reset(self):
        self.cursor = -1
        self.read_lines()
        self.label_files = []
        self.image_files = []
        self.epoch += 1
        self.current_line_no = -1

    def getpad(self):
        return 0

    def get_flist_name(self):
        return self.flist_name

    def read_lines(self):
        self.current_line_no = -1;
        with open(self.flist_name, 'r') as f:
            self.file_lines = f.readlines()
            if self.shuffle:
                self.random.shuffle(self.file_lines)
        f.close()

    def get_line(self):
        self.current_line_no += 1
        if (self.current_line_no < len(self.file_lines)):
            return self.file_lines[self.current_line_no]
        else:
            self.current_line_no = len(self.file_lines) - 1
            return self.file_lines[-1]


    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))

            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration

if __name__ == "__main__":
    test_FileIter()
