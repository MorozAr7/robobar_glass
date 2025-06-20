import albumentations as A
import random
import numpy as np
import cv2
import torch
import sys 
import os
from data_augmentation import data_aug
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_epoch_iters):
        self.n_epoch_iters = n_epoch_iters
        self.path_pos = "/home/morozart/RobobarGlassRecongition/ReducedSizeData/Positive/"
        self.path_neg = "/home/morozart/RobobarGlassRecongition/ReducedSizeData/Negative/"

        self.pos_files = os.listdir(self.path_pos)
        self.neg_files = os.listdir(self.path_neg)
        

        self.crop_bboxes = [[154, 277, 314, 437] ,
                        [294, 200, 460, 366] ,
                        [460, 122, 640, 302] ,
                        [653, 53, 833, 233] ,
                        [1567, 36, 1729, 198] ,
                        [1772, 89, 1934, 251] ,
                        [1972, 147, 2132, 307] ,
                        [2144, 209, 2308, 373] ,]

        self.inp_size_lims = (48, 128)
        self.bbox_rescale_lims = (0.5, 1.25)
        self.bbox_shift_lims = (-0.2, 0.2)
        self.n_pos = 25
        self.n_neg = 25
    

    def __len__(self):
        return self.n_epoch_iters

    def xyxy2cxcywh(self, xyxy):
        cx = (xyxy[0] + xyxy[2]) // 2
        cy = (xyxy[1] + xyxy[3]) // 2
        w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        return cx, cy, w, h

    def cxcywh2xyxy(self, cxcywh):
        cx, cy, w, h = cxcywh
        x_min, y_min = cx - w // 2, cy - h // 2
        x_max, y_max = cx + w // 2, cy + h // 2
        return x_min, y_min, x_max, y_max

    def crop_robust(self, img, bbox):
        x_min, y_min, x_max, y_max = bbox
        padding = [0, 0, 0, 0]
        if x_min < 0:
            padding[0] = -x_min
            x_min = 0
        if y_min < 0:
            padding[1] = -y_min
            y_min = 0
        if x_max > img.shape[1]:
            padding[2] = x_max - img.shape[1]
            
        if y_max > img.shape[0]:
            padding[3] = y_max - img.shape[0]
            
        # print(padding)
        if padding[0] != 0:
            img = np.vstack([np.zeros((padding[0], img.shape[1], 3), dtype=np.uint8), img])
        if padding[1] != 0:
            img = np.hstack([np.zeros((img.shape[0], padding[1], 3), dtype=np.uint8), img])
        if padding[2] != 0:
            img = np.vstack([img, np.zeros((padding[2], img.shape[1], 3), dtype=np.uint8)])
        if padding[3] != 0:
            img = np.hstack([img, np.zeros((img.shape[0], padding[3], 3), dtype=np.uint8)])

        return img[y_min:y_max, x_min:x_max]
        

    def crop_img(self, images_list, size_resize):
        crops = []

        for img in images_list:
            # print(iqmg.shape)
            for crop_bbox in self.crop_bboxes:
                cx, cy, w, h = self.xyxy2cxcywh(crop_bbox)
                rescale = random.uniform(*self.bbox_rescale_lims)
                w, h = int(w * rescale), int(h * rescale)
                shift_x = int(w * random.uniform(*self.bbox_shift_lims))
                shift_y = int(h * random.uniform(*self.bbox_shift_lims))

                cx, cy = cx + shift_x, cy + shift_y

                x_min, y_min, x_max, y_max = self.cxcywh2xyxy((cx, cy, w, h))
                
                
                crop = self.crop_robust(img, (x_min, y_min, x_max, y_max))
                crop = cv2.resize(crop, (size_resize, size_resize))
                crops.append(crop)
        return crops


    def load_image(self, path):
        return np.array(Image.open(path))
    
    def load_images_threads(self, files_list, path):
        with ThreadPoolExecutor(max_workers=len(files_list)) as executor:
            images = list(executor.map(self.load_image, [os.path.join(path, file) for file in files_list]))
        return images
            
    def __getitem__(self, _):
        pos_rands = random.sample(self.pos_files, self.n_pos)
        neg_rands = random.sample(self.neg_files, self.n_neg)

        size = random.randint(*self.inp_size_lims)
        
        pos_imgs = self.load_images_threads(pos_rands, self.path_pos)
        neg_imgs = self.load_images_threads(neg_rands, self.path_neg)

        pos_crops = self.crop_img(pos_imgs, size)
        neg_crops = self.crop_img(neg_imgs, size)

        labels = [1] * len(pos_crops) + [0] * len(neg_crops)

        imgs = pos_crops + neg_crops
        imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2) / 255.0

        labels = np.array(labels, dtype=np.float32).reshape(-1, 1)

        return imgs, labels
        

if __name__ == "__main__":
    dataset = Dataset()
    dataset.__getitem__(0)
        
