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
datasave_kaggle = os.path.abspath(os.path.join(BASE_DIR, "..", "GAN_RESULT", "2020_11_13"))
print(BASE_DIR)


for epoch in range(1):
    # fig = plt.figure(figsize=(20, 15))
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
    img_path = os.path.join(datasave_kaggle, "epoch_", str(epoch))
    if plot:
        plt.show()
    fig.savefig("imgs/epoch_" + epoch, dpi=200)

