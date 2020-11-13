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
import imageio
import h5py
import imageio
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as Tra

dim = 64
class Noisedataset(Dataset):

    def __init__(self, img_dir, data_augmentation_factor, mu, var, transforms=True, only_noise=False, name="train"):
        """
        Arguments:
            - img_dir 是上层目录
            - data_augmentation_factor: int - number of times an image is augmented. Set
            it to 0 if no augmentation is wanted.
            - mu: mean of the training set - used for standardization
            - var: standard deviation of the training set - used for standardization
            - transforms: bool - indicates whether to apply data transform
            - name: string - name of the dataset (train, validation or test)
            - only_noise: bool - whether or not to do residual learning
        """
        np.random.seed(0)

        if not data_augmentation_factor:
            self.data_augmentation_factor = 1
        else:
            self.data_augmentation_factor = data_augmentation_factor                                   # ata_augmentation_factor?
        self.transforms = transforms
        self.only_noise = only_noise
        self.name = name
        self.mean = mu
        self.std = var

        # Create dictionary that maps each image to some specific noise and the image's path
        dicts = []
        for root, _, files in os.walk(img_dir):
            for subfile in files:
                img_path = os.path.join(root, subfile)
                for _ in range(self.data_augmentation_factor):
                    # Define noise to be applied
                    p = np.random.rand()
                    if p < 1:
                        noise_type = "gaussian"
                    else:
                        noise_type = "speckle"
                    # Add image-noise pair information to the info dictionary
                    dicts.append({'path': img_path,
                                  'noise': noise_type})
        self.img_dict = {image_id: info for image_id, info in enumerate(
            dicts)}  #  self.img_dict: (0, {'path': 'C:\\Users\\Yan\\Documents\\LoDoPaB_CT_Dataset\\ground_truth_validation\\ground_truth_validation_000.hdf5',

    def __getitem__(self, image_id):
        np.random.seed(0)

        # read
        # img = h5py.File(self.img_dict[image_id]['path'], 'r')['data'].astype(float)  #class 'h5py._hl.dataset.Dataset'
        img = h5py.File(self.img_dict[image_id]['path'], 'r')['data']
        img_list = []
        for k in  range(len(img))
            a = np.array(img[k], torch.float64, copy = True)                   #（128,362,362）

        a_two = Tra.Resize(a, (dim, dim))
        img_tensor = torch.tensor(a_two)

        img_list.append(img[k])



        return img_tensor

    def __len__(self):
        return len(self.img_dict)
