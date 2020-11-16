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

# %%
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

for i in range(3):
    np.random.seed(0)
    fig = plt.figure(figsize=(20, 15))
    plt.imshow(img_list[i])
    plt.show()
    img_path = os.path.join(datasave_kaggle, "i_" + str(i))
    fig.savefig(img_path, dpi=200)



    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.axis('off')
    # ax2.axis('off')
    # ax3.axis('off')
    # ax1.set_title("Noisy image")
    # ax2.set_title("Denoised image")
    # ax3.set_title("Ground-truth")
    #
    # if only_noise:
    #     ax1.imshow(inputs, cmap='gray')
    #     ax2.imshow(inputs - outputs[0], cmap='gray')
    #     ax3.imshow(inputs - ground_truth[0], cmap='gray')
    # else:
    #     ax1.imshow(inputs, cmap='gray')
    #     ax2.imshow(outputs, cmap='gray')
    #     ax3.imshow(ground_truth, cmap='gray')


