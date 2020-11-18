import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import shutil
import matplotlib.pyplot as plt
from IPython import display
# from utils import Logger
# Pytorch Libraries
import torch
import torchvision as tv
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import zipfile   # no
import h5py
import copy
# import time      # no
import torchvision.models as models

# Modelisation Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob  # only glob2

import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

from IPython import display
# from utils import Logger
import torch
from torch import nn, optim
from torch import autograd
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from multiprocessing import freeze_support
import time


import gc;
gc.collect()

BASE_DIR = os.path.abspath('')
dataset_LDCT = os.path.abspath(os.path.join(BASE_DIR, "..", "LoDoPaB_CT_Dataset"))
datasave_kaggle = os.path.abspath(os.path.join(BASE_DIR, "..", "test_PIC", "2020_11_13"))
print(BASE_DIR)

# %% path
img_dir = os.path.join(dataset_LDCT, "ground_truth_validation")
name_list = list()                                          # validation images paths
for root, _, files in os.walk(img_dir):
    for subfile in files:
        img_path = os.path.join(root, subfile)
        name_list.append(img_path)

X_train = [h5py.File(i, 'r')['data'] for i in name_list]

img_dir_test = os.path.join(dataset_LDCT, "ground_truth_test")
name_list_test = list()                                     # test images paths
for root, _, files in os.walk(img_dir_test):
    for subfile in files:
        img_path = os.path.join(root, subfile)
        name_list_test.append(img_path)

X_test = [h5py.File(i, 'r')['data'] for i in name_list_test]
X_train = X_train + X_test

# %%
img_list = list()
for k in range(3):
    for i in range(128):
        img_list.append(X_train[k][i])
val_list = list()
for k in range(1):
    for i in range(128):
        val_list.append(X_train[k][i])


# transform
dim = 64
only_noise = False
img_transform = copy.deepcopy(img_list)
img_transform_show =list()
train_transform = transforms.Compose([
        # transforms.Resize((dim, dim)),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(0, 0.015),])


# standardize it

for img_i in img_transform:
    print(img_i.shape)
    # noise
    noise_type = "gaussian"
    if noise_type == "gaussian":
        noise = np.random.normal(0, 0.015, img_i.shape)
    # TODO: add the other noise
    img_i = img_i + noise
    # if residual learning, ground-truth should be the noise
    # TODO: add 'if only noise'
    img_i = train_transform(img_i)

    # TODO: show tensor image
    # FIXME: 3D TO 2D when in whole coding
    # TODO: tensor squeeze test one


# # plot
#
# for j in range(2):
#     fig = plt.figure(figsize=(20, 15))
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax2 = fig.add_subplot(1, 2, 2)
#     ax1.axis('off')
#     ax2.axis('off')
#     ax1.set_title("image")
#     ax2.set_title("image transform")
#     only_noise = True
#     if only_noise:
#         ax1.imshow(img_list[j], cmap='gray')
#         ax2.imshow(img_transform_show[j], cmap='gray')
#     plt.show()
#     img_path = os.path.join(datasave_kaggle, "test_one_dim64" + str(j))
#     fig.savefig(img_path, dpi=200)





# for i in range(3):
#     np.random.seed(0)
#     fig = plt.figure(figsize=(20, 15))
#     plt.imshow(img_list[i])
#     plt.show()
#     img_path = os.path.join(datasave_kaggle, "i_" + str(i))
#     fig.savefig(img_path, dpi=200)

a = torch.Tensor([[1.], [2.], [3.], [4.]])
#a = torch.Tensor([[[[1.], [2.], [3.], [4.]]]])
#a = torch.Tensor([1, 2, 3, 4])
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)


