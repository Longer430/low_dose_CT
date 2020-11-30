# -*- coding: utf-8 -*-
"""
# @file name  : Noisydataset.py
# @author     : Wenting LONG
# @date       : 2020-11-5
# @brief      : Dataset定义
# 1. Before Nov 12, I was trying to rewrite the dataloader, but something went wrong of the image size. I need to reset this.
"""

import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
import torchvision.transforms as T

# %%
class NoiseDataset(Dataset):

    def __init__(self, img_list, data_augmentation_factor, mu, var, dim, transforms=True, only_noise=False, name="train"):
        """
        Arguments:
            - img_list: list of the images paths
            - data_augmentation_factor: int - number of times an image is augmented. Set
            it to 0 if no augmentation is wanted.
            - mu: mean of the training set - used for standardization
            - var: standard deviation of the training set - used for standardization
            - transforms: bool - indicates whether to apply data augmentation
            - name: string - name of the dataset (train, validation or test)
            - only_noise: bool - whether or not to do residual learning
        """
        # np.random.seed(0)
        seed = 2
        torch.cuda.manual_seed(seed)
        if not data_augmentation_factor:
            self.data_augmentation_factor = 1
        else:
            self.data_augmentation_factor = data_augmentation_factor
        self.img_list = img_list
        self.transforms = transforms
        self.only_noise = only_noise
        self.name = name
        self.mean = mu
        self.std = var
        self.dim = dim

        # Create dictionary that maps each image to some specific noise and the
        # image's path
        dicts = []
        for image_path in img_list:
            for _ in range(self.data_augmentation_factor):

                # Define noise to be applied
                p = np.random.rand()
                if p < 1:
                    noise_type = "gaussian"
                else:
                    noise_type = "speckle"

                # Add image-noise pair information to the info dictionary
                dicts.append({'path': image_path,
                              'noise': noise_type})
        self.img_dict = {image_id: info for image_id, info in enumerate(dicts)}

    def __getitem__(self, image_id):
        # np.random.seed(0)
        seed = 2
        torch.cuda.manual_seed(seed)
        # load image
        # img = imageio.imread(self.img_dict[image_id]['path']).astype(np.uint8)
        img = self.img_dict[image_id]['path']
        # standardize it
        #img = (img - img.min()) /(img.max() - img.min())
        # downsample the images' size (to speed up training)
        img = resize(img, (self.dim, self.dim))

        # create noisy image
        noise_type = self.img_dict[image_id]['noise']
        if noise_type == "gaussian":
            noise = np.random.normal(0, 0.015, img.shape)
        elif noise_type == "speckle":
            noise = img * np.random.randn(img.shape[0], img.shape[1]) / 5
        noisy_img = img + noise
        noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())
        # if residual learning, ground-truth should be the noise
        if self.only_noise:
            img = noise

        # convert to PIL images
        img = Image.fromarray(img)
        noisy_img = Image.fromarray(noisy_img)

        # apply the same data augmentation transformations to the input and the
        # ground-truth
        p = np.random.rand()
        if self.transforms and p < 0.5:
            self.t = T.Compose([T.RandomHorizontalFlip(1), T.ToTensor()])
        else:
            self.t = T.Compose([T.ToTensor()])
        img = self.t(img)
        noisy_img = self.t(noisy_img)
        return noisy_img, img

    def __len__(self):
        return len(self.img_dict)


