#!/usr/bin/env python
# coding: utf-8
"""
# @file name  : Noisydataset.py
# @author     : Wenting LONG
# @date       : 2020-11-5
# @brief      : Dataset定义
# 1. Before Nov 12, I was trying to rewrite the dataloader, but something went wrong of the image size. I need to reset this.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import h5py
import time
# from IPython import display
# from utils import Logger

# Pytorch Libraries

# import torchvision as tv
# import torchvision.transforms as T
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import zipfile   # no

# import time      # no
# import torchvision.models as models
# Modelisation Libraries


# from tqdm import tqdm
# from glob import glob    # only glob2


# import imageio

# from skimage.transform import resize
#
# from IPython import display
# # from utils import Logger
# import torch
# from torch import nn, optim
# from torch import autograd
# from torch.autograd.variable import Variable
# from torchvision import transforms, datasets


from Noisedataset import Noisedataset
# import Discriminator
# import FeatureExtractor
# import Generator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# parameters
dim = 64  # where?
out_shape = 8  # not exist

Lambda = 10
batch_size = 10
num_epochs = 1

##################################################################################
# get_ipython().run_line_magic('load_ext', 'autoreload')   # what is this?
# get_ipython().run_line_magic('matplotlib', 'inline')
##################################################################################

# ============================ step 1/5 data  ============================
BASE_DIR = os.path.abspath('')
dataset_LDCT = os.path.abspath(os.path.join(BASE_DIR, "..", "LoDoPaB_CT_Dataset"))
img_dir = os.path.join(dataset_LDCT, "ground_truth_test")
val_dir = os.path.join(dataset_LDCT, "ground_truth_validation")


train = Noisedataset(img_dir, 1, 0, 1, only_noise = False, name="train")
val = Noisedataset(val_dir, 1, 0, 1, only_noise = False, name="validation")

torch.manual_seed(1)
data_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val, batch_size=1, shuffle=False)


for i in enumerate(data_loader):
    pass
# ============================ step 2/5 generator ============================

# generator = Generator()
# discriminator = Discriminator()
# if torch.cuda.is_available():
#     generator.cuda()
#     discriminator.cuda()
#     features_extractor.cuda()
# d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.9))
# g_optimizer = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
#
#
# ============================ step 3/5 loss function ============================
#
#
# def vgg_loss(pred, gt, features_extractor = features_extractor):
#   #vgg_pred = features_extractor(pred)
#   vgg_gt = features_extractor(gt)
#   vgg_pred = features_extractor(pred)
#   size = vgg_gt.shape
# #   normalized = 1/(size[1]*size[2]*size[3])
#   mse = nn.MSELoss()
#   return mse(vgg_pred, vgg_gt)
#
#
# # ============================ step 4/5 discriminator ============================
#
#
# def calc_gradient_penalty(discriminator, real_data, fake_data, Lambda, batch_size=16):
#
#     alpha = torch.rand(real_data.shape[0], 1).cuda()
#     alpha = alpha.expand(real_data.shape[0], int(real_data.nelement()/real_data.shape[0])).contiguous()
#     alpha = alpha.view(real_data.shape[0], 1, dim, dim)
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#     interpolates = interpolates.requires_grad_(True)
#
#     disc_interpolates = discriminator(interpolates)
#
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
#     return gradient_penalty

# # ============================ step 5/5 train ============================
# def train_discriminator_generator(optimizer_d, optimizer_g, real_data, input_img,
#                                   batch_size, loss_function, Lambda=10, lambda_2=10):
#
#    # real_data = Normal DCT & input_img = Low DTC
#     #1. Train Discriminator
#     #Reset gradients
#     fake_data = generator(input_img)
#     for k in range(2) :
#       optimizer_d.zero_grad()
#       #1.1 Train on Real Data
#       prediction_real = discriminator(real_data)
#       # Calculate error and backpropagate
#       error_real = -torch.mean(prediction_real)
#
#       # 1.2 Train on Fake Data
#       prediction_fake = discriminator(fake_data.detach())
#       # Calculate error and backpropagate
#       error_fake = torch.mean(prediction_fake)
#       gradient_penalty = calc_gradient_penalty(discriminator, real_data,
#                                               fake_data.detach(), Lambda=Lambda,
#                                               batch_size=batch_size)
#
#       loss_discriminator = error_real + error_fake
#       loss_discriminator.backward()
#       gradient_penalty.backward()
#
#       # 1.3 Update weights with gradients
#       optimizer_d.step()
#
#     # 2. Train Generator
#     # Reset gradients
#     optimizer_g.zero_grad()
#     # generate fake data
# #     fake_data = generator(input_img)
#     prediction_img = discriminator(fake_data)
#     # Calculate error and backpropagate
#     loss_generator = -torch.mean(prediction_img)
#     loss_vgg = lambda_2 * loss_function(fake_data, real_data)
#     loss_generator_t = loss_generator + loss_vgg
#     loss_generator_t.backward()
#     # Update weights with gradients
#     optimizer_g.step()
#
#     # print(loss_discriminator, loss_generator, loss_vgg, gradient_penalty)
#     return loss_discriminator, loss_generator, loss_vgg, gradient_penalty, error_real.mean(), error_fake.mean(), fake_data

# def validate_discriminator_generator(real_data, input_img,
#                                     batch_size, loss_function,
#                                      Lambda=10, lambda_2=10):
#
#    # real_data = Normal DCT & input_img = Low DTC
#     fake_data = generator(input_img)
#     prediction_real = discriminator(real_data)
#     error_real = -torch.mean(prediction_real)
#     prediction_fake = discriminator(fake_data.detach())
#     error_fake = torch.mean(prediction_fake)
#     gradient_penalty = calc_gradient_penalty(discriminator, real_data,
#                                               fake_data.detach(), Lambda=Lambda,
#                                               batch_size=batch_size)
#
#     loss_discriminator = error_real + error_fake
#     loss_generator = - error_fake
#     loss_vgg = lambda_2 * loss_function(fake_data, real_data)
#
#     return loss_discriminator, loss_generator, loss_vgg, gradient_penalty, error_real.mean(), error_fake.mean(), fake_data
#
#
# ============================ step 5/5  train ============================
#
#
# for n_batch, (test_images, real_test_img) in enumerate(val_data_loader) :
#     if n_batch ==1:
#         break
# input_test_img = test_images.cuda()
# real_test_img = real_test_img.cuda()
#
#
# # In[ ]:
#
#
# def plot_prediction(inputs, outputs, ground_truth, only_noise, plot=False, epoch='0'):
#     size = inputs.shape
#     inputs = np.vstack(inputs.data.cpu().numpy().reshape(size[0], size[2], size[3]))
#     outputs = np.vstack(outputs.data.cpu().numpy().reshape(size[0], size[2], size[3]))
#     ground_truth = np.vstack(ground_truth.data.cpu().numpy().reshape(size[0], size[2], size[3]))
#     for i in range(1):
#         fig = plt.figure(figsize=(20, 15))
#         ax1 = fig.add_subplot(1, 3, 1)
#         ax2 = fig.add_subplot(1, 3, 2)
#         ax3 = fig.add_subplot(1, 3, 3)
#         ax1.axis('off')
#         ax2.axis('off')
#         ax3.axis('off')
#         ax1.set_title("Noisy image")
#         ax2.set_title("Denoised image")
#         ax3.set_title("Ground-truth")
#
#         if only_noise:
#             ax1.imshow(inputs, cmap='gray')
#             ax2.imshow(inputs - outputs[0], cmap='gray')
#             ax3.imshow(inputs - ground_truth[0], cmap='gray')
#         else:
#             ax1.imshow(inputs, cmap='gray')
#             ax2.imshow(outputs, cmap='gray')
#             ax3.imshow(ground_truth, cmap='gray')
#         if plot: plt.show()
#         fig.savefig("images/epoch_" + epoch, dpi = 200)
#
# def display_status(epoch, num_epochs, n_batch,
#                     p_real, loss_vgg, p_fake, penalty,
#                     batch_ = int(len(data_loader.dataset)/batch_size)) :
#   print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
#   epoch, num_epochs, n_batch, batch_))
#   print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
#   print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(p_real, p_fake))
#   print('Loss VGG : {:.4f}'.format(loss_vgg.data.cpu()))
#   print('Gradient penalty : {:.4f}'.format(penalty.data.cpu()))
#
#
# # In[ ]:
#
#
# import gc; gc.collect()
#
#
# # In[ ]:
#
#
# def MSE(image_true, image_generated):
#     return ((image_true - image_generated) ** 2).mean()
#
#
# def PSNR(image_true, image_generated):
#     mse = MSE(image_true, image_generated)
#     return -10 * torch.log10(mse)
#
#
# def SSIM(image_true, image_generated, C1=0.01, C2=0.03):
#     mean_true = image_true.mean()
#     mean_generated = image_generated.mean()
#     std_true = image_true.std()
#     std_generated = image_generated.std()
#     covariance = ((image_true - mean_true) * (image_generated - mean_generated)).mean()
#
#     numerator = (2 * mean_true * mean_generated + C1) * (2 * covariance + C2)
#     denominator = (mean_true ** 2 + mean_generated ** 2 + C1) * (std_true ** 2 + std_generated ** 2 + C2)
#     return numerator / denominator
#
#
# # sd
#
# # ## Entraînement du modèle
#
# # In[ ]:
#
#
# loss_function = {'d_error' : {'train' : [], 'test' : []},
#                  'g_error' : {'train' : [], 'test' : []},
#                  'loss_vgg' : {'train' : [], 'test' : []},
#                  'penalty' : {'train' : [], 'test' : []},
#                  'p_real' : {'train' : [], 'test' : []},
#                  'p_fake' : {'train' : [], 'test' : []},
#                 'PSNR' : {'train' : [], 'test' : []},
#                 'SSIM' : {'train' : [], 'test' : []}}
# def update_loss_function(d_error, g_error, loss_vgg,
#                          penalty , p_real, p_fake, psnr, ssim, mode) :
#     loss_function['d_error'][mode].append(d_error.data.cpu().numpy().reshape(1)[0])
#     loss_function['g_error'][mode].append(g_error.data.cpu().numpy().reshape(1)[0])
#     loss_function['loss_vgg'][mode].append(loss_vgg.data.cpu().numpy().reshape(1)[0])
#     loss_function['penalty'][mode].append(penalty.data.cpu().numpy().reshape(1)[0])
#     loss_function['p_real'][mode].append(-p_real.data.cpu().numpy().reshape(1)[0])
#     loss_function['p_fake'][mode].append(p_fake.data.cpu().numpy().reshape(1)[0])
#     loss_function['PSNR'][mode].append(psnr.data.cpu().numpy().reshape(1)[0])
#     loss_function['SSIM'][mode].append(ssim.data.cpu().numpy().reshape(1)[0])
#
#
# # In[ ]:
#
#
# name = '_wgan_vgg'
#
#
# # In[ ]:
#
#
# for epoch in range(num_epochs):
#     for n_batch, (input_img, real_data) in enumerate(data_loader):
#         real_data = Variable(real_data)
#         if torch.cuda.is_available(): real_data = real_data.cuda()
#         if torch.cuda.is_available(): input_img = input_img.cuda()
#         # Train D & G
#         d_error, g_error, loss_vgg, penalty , p_real, p_fake, pred_images = train_discriminator_generator(d_optimizer,
#                                                          g_optimizer,
#                                                          real_data,
#                                                          input_img,
#                                                          loss_function=vgg_loss,
#                                                          batch_size=batch_size)
#
# #         pred_images = generator(input_img)
# #         g_optimizer.zero_grad()
# #         loss_vgg = nn.MSELoss()(pred_images, real_data); loss_vgg.backward()
# #         g_optimizer.step()
#         # Display Progress
#         if (n_batch) % 120  == 0:
#             display.clear_output(False)
#             # Display Images
#             psnr = PSNR(real_data, pred_images)
#             ssim = SSIM(real_data, pred_images)
#             update_loss_function(d_error, g_error, loss_vgg,
#                          penalty , p_real, p_fake, psnr, ssim, mode='train')
#             d_error, g_error, loss_vgg_test, penalty , p_real, p_fake, pred_images = validate_discriminator_generator(real_test_img, input_test_img,
#                                                                           batch_size=2, loss_function=vgg_loss)
# #             pred_images = generator(input_test_img)
# #             loss_vgg_test = nn.MSELoss()(real_test_img, pred_images)
#             psnr = PSNR(real_test_img, pred_images)
#             ssim = SSIM(real_test_img, pred_images)
#             update_loss_function(d_error, g_error, loss_vgg_test,
#                          penalty, p_real, p_fake, psnr, ssim, mode='test')
#
# #             pred_images = generator(input_test_img.cuda()).data.cpu()
#             display_status(epoch, num_epochs + 10, n_batch,
#                     p_real, loss_vgg, p_fake, penalty,
#                     batch_ = int(len(data_loader.dataset)/batch_size))
#             if epoch < 100 :
#               epoch_ = '0' + str(epoch)
#             else :
#                 epoch_ = epoch
#             plot_prediction(input_test_img, pred_images, real_test_img,
#                             only_noise, plot=True,
#                             epoch= str(epoch_) + "_" + str(n_batch))
#             gc.collect()
#
#
# # In[ ]:
#
#
# fig, ax = plt.subplots(4, 2, figsize =(30, 20))
# ax = ax.flatten()
# i = 0
# for key, values in loss_function.items() :
# #     if key in ['SSIM', 'loss_vgg', 'PSNR']:
#     ax[i].plot(values['test'][2:], label='test')
#     ax[i].plot(values['train'][2:], label='train')
#     ax[i].set_title(key)
#     ax[i].legend(loc='best')
#     i = i+1
#
#
# # In[ ]:
#
#
# torch.save(generator.state_dict(), 'generator'+ name)
# torch.save(discriminator.state_dict(), 'discriminator' + name)
# np.save(name, loss_function)
# fig.savefig('loss_gen' + name +'.png')
#
#
# # # Evaluation des différents modèles
#
# # In[ ]:
#
#
# path_models = ['generator_alone_mse', 'generator_alone_vgg',
#                'generator_wgan_mse', 'generator_wgan_vgg']
# dico_metrics = {elt : {'ssim' : [], 'psnr' : []} for elt in path_models}
#
# def evaluation(path, data_loader, discriminator = None) :
#     generator.load_state_dict(torch.load('olds_models/'  + path))
#     generator.eval()
#     print("Testing model :", path)
#     for n_batch, (input_img, real_data) in enumerate(data_loader):
#         real_data = Variable(real_data)
#         if torch.cuda.is_available(): real_data = real_data.cuda()
#         if torch.cuda.is_available(): input_img = input_img.cuda()
#         pred_images = generator(input_img)
#         psnr = PSNR(real_data, pred_images).data.cpu().numpy().reshape(1)[0]
#         ssim = SSIM(real_data, pred_images).data.cpu().numpy().reshape(1)[0]
#         dico_metrics[path]['ssim'].append(ssim)
#         dico_metrics[path]['psnr'].append(psnr)
#     #plot_prediction(input_img[:3], pred_images[:3], real_data[:3],
#     #            only_noise, plot=True,
#     #            epoch=path)
#     return input_img[-1], pred_images[-1], real_data[-1]
#
# to_plot = []
# for path in path_models :
#     to_plot.append(evaluation(path, val_data_loader, discriminator = None))
#
#
# # ## Visualisation of the output of our models on a test image
#
# # In[ ]:
#
#
# fig, axes = plt.subplots(2, 3, figsize = (15, 10))
# axes[0, 0].imshow(to_plot[0][0].data.cpu().numpy().reshape(64, 64), cmap='gray')
# axes[0, 1].imshow(to_plot[0][-1].data.cpu().numpy().reshape(64, 64), cmap='gray')
# axes[0, 2].imshow(to_plot[0][1].data.cpu().numpy().reshape(64, 64), cmap='gray')
# axes[1, 0].imshow(to_plot[1][1].data.cpu().numpy().reshape(64, 64), cmap='gray')
# axes[1, 1].imshow(to_plot[2][1].data.cpu().numpy().reshape(64, 64), cmap='gray')
# axes[1, 2].imshow(to_plot[3][1].data.cpu().numpy().reshape(64, 64), cmap='gray')
#
# axes[0, 0].set_title('Noisy image', fontsize=18)
# axes[0, 1].set_title('Ground truth image', fontsize=18)
# axes[0, 2].set_title('Generator-MSE', fontsize=18)
# axes[1, 0].set_title('Generator-VGG', fontsize=18)
# axes[1, 1].set_title('WGAN-MSE', fontsize=18)
# axes[1, 2].set_title('WGAN-VGG', fontsize=18)
# axes = axes.flatten()
# for i in range (len(axes)) :
#     axes[i]
#     axes[i].axis('off')
# fig.savefig('image_test', dpi = 100, bbox_inches='tight')
#
#
# # In[ ]:
#
#
# import pandas as pd
# df = pd.DataFrame.from_dict(dico_metrics)
# for col in df.columns :
#     df[col] = df[col].apply(lambda x: x[-1])
#     df.T
#
#
# # ### Computing the mean of our metrics on the validation data
#
# # In[ ]:
#
#
# import pandas as pd
# df = pd.DataFrame.from_dict(dico_metrics)
# for col in df.columns :
#     df[col] = df[col].apply(lambda x: np.mean(x))
# df.T
#
#
# # ## Loss function over epochs
#
# # In[ ]:
#
#
# loss_gen_alone_mse = np.load('_alone_mse.npy', allow_pickle='TRUE').item()['loss_vgg']['train']
# loss_gen_alone_vgg = np.load('_alone_vgg.npy', allow_pickle='TRUE').item()['loss_vgg']['train']
# loss_gen_wgan_mse = np.load('_wgan_mse.npy', allow_pickle='TRUE').item()['loss_vgg']['train']
# loss_gen_wgan_vgg = np.load('_wgan_vgg.npy', allow_pickle='TRUE').item()['loss_vgg']['train']
# loss_disc_wgan_mse = np.load('_wgan_mse.npy', allow_pickle='TRUE').item()
# loss_disc_wgan_vgg = np.load('_wgan_vgg.npy', allow_pickle='TRUE').item()
#
#
# # In[ ]:
#
#
# import matplotlib.gridspec as gridspec
# gs = gridspec.GridSpec(2, 2)
#
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
# ax1.plot(loss_gen_alone_mse[:120], label = 'Generator-MSE : MSE-Loss')
# ax1.plot(10*np.array(loss_gen_wgan_mse)[:120], label = 'WGAN-MSE : MSE-Loss')
# ax1.legend(loc='best')
# ax1.set_xlabel('epoch');
#
# ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
# ax2.plot(loss_gen_alone_vgg[:120], label = 'Generator-VGG : VGG-Loss')
# ax2.plot(10*np.array(loss_gen_wgan_vgg)[:120], label = 'WGAN-VGG : VGG-Loss')
# ax2.legend(loc='best')
# ax2.set_xlabel('epoch');
#
# ax3 = fig.add_subplot(gs[1, 0]) # row 1, span all columns
# ax3.plot(loss_disc_vgg['p_real']['train'][:130], label = 'WGAN-MSE : D(x)')
# ax3.plot(loss_disc_wgan['p_real']['train'][:130], label = 'WGAN-VGG : D(x)')
# ax3.legend(loc='best')
# ax3.set_xlabel('epoch');
#
# ax4 = fig.add_subplot(gs[1, 1]) # row 1, span all columns
# ax4.plot(loss_disc_wgan_mse['p_fake']['train'][:130], label = 'WGAN-MSE : D(G(x))')
# ax4.plot(loss_disc_wgan_vgg['p_fake']['train'][:130], label = 'WGAN-VGG : D(G(x))')
# ax4.legend(loc='best')
# ax4.set_xlabel('epoch');
#
#
# # In[ ]:
#
#
# fig.savefig('all_loss.png', dpi =100, bbox_inches='tight')
#
#
# # In[ ]:
#
#
# loss_disc_vgg_al = np.load('_alone_vgg.npy', allow_pickle='TRUE').item()
# loss_disc_mse_al = np.load('_alone_mse.npy', allow_pickle='TRUE').item()
#
#
# # In[ ]:
#
#
# fig, axes = plt.subplots(2, 1, figsize = (15, 10))
# axes[0].plot(loss_disc_wgan_mse['PSNR']['test'][:130], label = 'WGAN-MSE')
# axes[0].plot(loss_disc_wgan_vgg['PSNR']['test'][:130], label = 'WGAN-VGG')
# axes[0].plot(loss_disc_vgg_al['PSNR']['train'][:130], label = 'Generator-VGG')
# axes[0].plot(loss_disc_mse_al['PSNR']['train'][:130], label = 'Generator-MSE')
#
# axes[1].plot(loss_disc_wgan_mse['SSIM']['test'][:130], label = 'WGAN-MSE')
# axes[1].plot(loss_disc_wgan_vgg['SSIM']['test'][:130], label = 'WGAN-VGG')
# axes[1].plot(loss_disc_vgg_al['SSIM']['train'][:130], label = 'Generator-VGG')
# axes[1].plot(loss_disc_mse_al['SSIM']['train'][:130], label = 'Generator-MSE')
# axes[0].set_title('PSNR over epochs')
# axes[1].set_title('SSIM over epochs')
# axes[0].legend(loc='best')
# axes[1].legend(loc='best')
#
#
# # In[ ]:
#
#
# fig.savefig('metrics_epochs.png', dpi =100, bbox_inches='tight')
#
#
# # In[ ]:
