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

# TODO: what is this?
# TODO: What else need to delete?

dim = 64
class Noisedataset(Dataset):

    def __init__(self, only_noise, data_augmentation_factor, img_list, num_works, transform = None,  name="train"):
        # TODO: only_noise=False, data_augmentation_factor?
        """
        Arguments:
            - img_list: list of the images paths
            it to 0 if no augmentation is wanted
            - transforms: bool - indicates whether to apply data augmentation
            - name: string - name of the dataset (train, validation or test)
            - only_noise: bool - whether or not to do residual learning
            - data_augmentation_factor: int - number of times an image is augmented. Set
        """
        np.random.seed(0)

        if not data_augmentation_factor:
            self.data_augmentation_factor = 1
        else:
            self.data_augmentation_factor = data_augmentation_factor
        self.img_list = img_list
        self.transforms = transform
        self.only_noise = only_noise
        self.name = name
        self.num_works = num_works

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
        np.random.seed(0)
        img = self.img_dict[image_id]['path'].astype(np.uint8)
        img = resize(img, (dim, dim))  # transform
        noise_type = self.img_dict[image_id]['noise']
        # TODO: down sample here or in the end
        # create noisy image
        if noise_type == "gaussian":
            noise = np.random.normal(0, 0.015, img.shape)
        elif noise_type == "speckle":
            noise = img * np.random.randn(img.shape[0], img.shape[1]) / 5
        noisy_img = img + noise
        noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())
        # if residual learning, ground-truth should be the noise
        if self.only_noise:
            img = noise

        # # convert to PIL images
        img = Image.fromarray(img)
        noisy_img = Image.fromarray(noisy_img)

        if self.transforms is not None:
            img = self.transforms(img)
            noisy_img = self.transforms(noisy_img)

        return noisy_img, img

    def __len__(self):
        return len(self.img_dict)
