import kornia.augmentation as K
import kornia.geometry.transform as KT
import torch 
import random


transform_normalize = K.AugmentationSequential(
    K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    same_on_batch=True
)


transform_augment = K.AugmentationSequential(
    K.RandomVerticalFlip(p=0.5),
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=1.0),  
    K.RandomGaussianBlur(kernel_size=(3, 11), sigma=(0.1, 2.0), p=1.0),
    same_on_batch=False,
)

    

