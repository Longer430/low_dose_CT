# %%
"""
# DLMI PROJECT : Low-Dose CT Image Denoising

# # @file name  : Noisydataset.py
# # @author     : Wenting LONG
# # @date       : 2020-11-5
# # @brief      : Dataset定义
# # 1. I only delete the .cuda() and reset the reading path and this has 3 problems.
A.
"""

# %%
# General Libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

# figure
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display
from glob import glob  # only glob2
from multiprocessing import freeze_support


#
from tqdm import tqdm
import time
import progressbar
p = progressbar.ProgressBar()
import gc;
gc.collect()

# cuda
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled = False

# class
from Generator import Generator
from Noisedataset import NoiseDataset
from Discriminator import Discriminator
from FeatureExtractor import FeatureExtractor

# NO USE
# import random
# import shutil
# import imageio
# from torch.autograd.variable import Variable
# import torch.nn.functional as F
#
# from skimage.transform import resize
# from utils import Logger

# %%
dim = 64
BASE_DIR = os.path.abspath('')
dataset_LDCT = os.path.abspath(os.path.join(BASE_DIR, "..", "LoDoPaB_CT_Dataset"))
datasave_kaggle = os.path.abspath(os.path.join(BASE_DIR, "..", "Lowdose_program", "GAN_RESULT", "2020_11_13"))
print(dataset_LDCT)
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
for k in range(49):
    for i in range(128):
        img_list.append(X_train[k][i])
val_list = list()
for k in range(49, 54):
    for i in range(128):
        val_list.append(X_train[k][i])
print(len(val_list))
print(len(img_list))

# %%
plt.imshow(X_train[0][26])


features_extractor = FeatureExtractor()
for param in features_extractor.parameters():
    param.requires_grad = False

# %%
Lambda = 10
batch_size = 16   # 32 will make storage run out of storage
num_epochs = 1

# %%
only_noise = False
train = NoiseDataset(img_list, 1, 0, 1, dim, False, only_noise, name="train")
val = NoiseDataset(val_list, 1, 0, 1, dim, False, only_noise, name="validation")

# %%
torch.cuda.manual_seed(1)
data_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=1)

val_data_loader = torch.utils.data.DataLoader(
    val, batch_size=2, shuffle=False, num_workers=1)

# %%
generator = Generator()
discriminator = Discriminator()
if torch.cuda.is_available():
    generator.to(device)
    discriminator.to(device)
    features_extractor.to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.9))
g_optimizer = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))


if __name__ == '__main__':
    freeze_support()
# %%
    def vgg_loss(pred, gt, features_extractor=features_extractor):
        # vgg_pred = features_extractor(pred)
        vgg_gt = features_extractor(gt)
        vgg_pred = features_extractor(pred)
        size = vgg_gt.shape
        #   normalized = 1/(size[1]*size[2]*size[3])
        mse = nn.MSELoss()
        return mse(vgg_pred, vgg_gt)


# %%
    def calc_gradient_penalty(discriminator, real_data, fake_data, Lambda, batch_size=16):
        if torch.cuda.is_available():
            real_data = real_data.to(device)
            fake_data = fake_data.to(device)
        alpha_cpu = torch.rand(real_data.shape[0], 1)
        alpha = alpha_cpu.to(device)
        #alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], int(real_data.nelement() / real_data.shape[0])).contiguous()
        alpha = alpha.view(real_data.shape[0], 1, dim, dim)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        grad_outputs_cpu = torch.ones(disc_interpolates.size())
        grad_outputs_gpu = grad_outputs_cpu.to(device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs = grad_outputs_gpu,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        # TODO: need to rewrite
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
        return gradient_penalty


    def train_discriminator_generator(optimizer_d, optimizer_g, real_data, input_img,
                                  batch_size, loss_function, Lambda=10, lambda_2=10):
        # real_data = Normal DCT & input_img = Low DTC
        # 1. Train Discriminator
        # Reset gradients
        if torch.cuda.is_available():
            real_data = real_data.to(device)
            input_img = input_img.to(device)
        fake_data = generator(input_img)
        for k in range(2):
            optimizer_d.zero_grad()
            # 1.1 Train on Real Data
            prediction_real = discriminator(real_data)
            # Calculate error and backpropagate
            error_real = -torch.mean(prediction_real)

            # 1.2 Train on Fake Data
            prediction_fake = discriminator(fake_data.detach())
            # Calculate error and backpropagate
            error_fake = torch.mean(prediction_fake)
            gradient_penalty = calc_gradient_penalty(discriminator, real_data,
                                                 fake_data.detach(), Lambda=Lambda,
                                                 batch_size=batch_size)

            loss_discriminator = error_real + error_fake
            loss_discriminator.backward()
            gradient_penalty.backward()

            # 1.3 Update weights with gradients
            optimizer_d.step()

            # 2. Train Generator
            # Reset gradients
            optimizer_g.zero_grad()
            # generate fake data
            #     fake_data = generator(input_img)
            prediction_img = discriminator(fake_data)
            # Calculate error and backpropagate
            loss_generator = -torch.mean(prediction_img)
            loss_vgg = lambda_2 * loss_function(fake_data, real_data)
            loss_generator_t = loss_generator + loss_vgg
            loss_generator_t.backward()
            # Update weights with gradients
            optimizer_g.step()

            # print(loss_discriminator, loss_generator, loss_vgg, gradient_penalty)
            return loss_discriminator, loss_generator, loss_vgg, gradient_penalty, error_real.mean(), error_fake.mean(), fake_data


    def validate_discriminator_generator(real_data, input_img,
                                     batch_size, loss_function,
                                     Lambda=10, lambda_2=10):
        # real_data = Normal DCT & input_img = Low DTC
        if torch.cuda.is_available():
            real_data = real_data.to(device)
        if torch.cuda.is_available():
            input_img = input_img.to(device)
        fake_data = generator(input_img)
        prediction_real = discriminator(real_data)
        error_real = -torch.mean(prediction_real)
        prediction_fake = discriminator(fake_data.detach())
        error_fake = torch.mean(prediction_fake)
        gradient_penalty = calc_gradient_penalty(discriminator, real_data,
                                             fake_data.detach(), Lambda=Lambda,
                                             batch_size=batch_size)

        loss_discriminator = error_real + error_fake
        loss_generator = - error_fake
        loss_vgg = lambda_2 * loss_function(fake_data, real_data)

        return loss_discriminator, loss_generator, loss_vgg, gradient_penalty, error_real.mean(), error_fake.mean(), fake_data


# %%
    for n_batch, (test_images, real_test_img) in enumerate(val_data_loader):
        if n_batch == 1:
            break
        input_test_img = test_images
        real_test_img = real_test_img

        if torch.cuda.is_available():
            input_test_img = input_test_img.to(device)
            real_test_img = real_test_img.to(device)
            try:
                output = nn.MSELoss(input)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception


# %%
    def plot_prediction(inputs, outputs, ground_truth, only_noise, plot=False, epoch='0'):
        size = inputs.shape
        inputs = np.vstack(inputs.data.cpu().numpy().reshape(size[0], size[2], size[3]))
        outputs = np.vstack(outputs.data.cpu().numpy().reshape(size[0], size[2], size[3]))
        ground_truth = np.vstack(ground_truth.data.cpu().numpy().reshape(size[0], size[2], size[3]))
        for i in range(1):
            fig = plt.figure(figsize=(20, 15))
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax1.set_title("Noisy image")
            ax2.set_title("Denoised image")
            ax3.set_title("Ground-truth")

            if only_noise:
                ax1.imshow(inputs, cmap='gray')
                ax2.imshow(inputs - outputs[0], cmap='gray')
                ax3.imshow(inputs - ground_truth[0], cmap='gray')
            else:
                ax1.imshow(inputs, cmap='gray')
                ax2.imshow(outputs, cmap='gray')
                ax3.imshow(ground_truth, cmap='gray')
            if plot:
                plt.show()
            img_save_path = os.path.join(datasave_kaggle, "epoch_" + str(epoch))
            fig.savefig(img_save_path, dpi=300)


    def display_status(epoch, num_epochs, n_batch,
                    p_real, loss_vgg, p_fake, penalty,
                    batch_=int(len(data_loader.dataset) / batch_size)):
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, batch_))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(p_real, p_fake))
        print('Loss VGG : {:.4f}'.format(loss_vgg.data.to("cpu")))
        print('Gradient penalty : {:.4f}'.format(penalty.data.to("cpu")))


    def MSE(image_true, image_generated):
        return ((image_true - image_generated) ** 2).mean()


    def PSNR(image_true, image_generated):
        mse = MSE(image_true, image_generated)
        return (-10 * torch.log10(mse))


    def SSIM(image_true, image_generated, C1=0.01, C2=0.03):
        mean_true = image_true.mean()
        mean_generated = image_generated.mean()
        std_true = image_true.std()
        std_generated = image_generated.std()
        covariance = ((image_true - mean_true) * (image_generated - mean_generated)).mean()

        numerator = (2 * mean_true * mean_generated + C1) * (2 * covariance + C2)
        denominator = (mean_true ** 2 + mean_generated ** 2 + C1) * (std_true ** 2 + std_generated ** 2 + C2)
        return numerator / denominator

    loss_function = {'d_error': {'train': [], 'test': []},
                        'g_error': {'train': [], 'test': []},
                        'loss_vgg': {'train': [], 'test': []},
                        'penalty': {'train': [], 'test': []},
                        'p_real': {'train': [], 'test': []},
                        'p_fake': {'train': [], 'test': []},
                        'PSNR': {'train': [], 'test': []},
                        'SSIM': {'train': [], 'test': []}}

    def update_loss_function(d_error, g_error, loss_vgg,
                                 penalty, p_real, p_fake, psnr, ssim, mode):
            loss_function['d_error'][mode].append(d_error.data.cpu().numpy().reshape(1)[0])
            loss_function['g_error'][mode].append(g_error.data.cpu().numpy().reshape(1)[0])
            loss_function['loss_vgg'][mode].append(loss_vgg.data.cpu().numpy().reshape(1)[0])
            loss_function['penalty'][mode].append(penalty.data.cpu().numpy().reshape(1)[0])
            loss_function['p_real'][mode].append(-p_real.data.cpu().numpy().reshape(1)[0])
            loss_function['p_fake'][mode].append(p_fake.data.cpu().numpy().reshape(1)[0])
            loss_function['PSNR'][mode].append(psnr.data.cpu().numpy().reshape(1)[0])
            loss_function['SSIM'][mode].append(ssim.data.cpu().numpy().reshape(1)[0])



# %%
    name = '_wgan_vgg'

# %%
    for epoch in tqdm(range(num_epochs), desc= "Plot"):
        time.sleep(0.5)
        # for n_batch, (input_img, real_data) in tqdm(enumerate(data_loader), desc="Learning"):
        # time.sleep(0.01)
        for n_batch, (input_img, real_data) in enumerate(data_loader):

            input_img = torch.tensor(input_img)
            real_data = torch.tensor(real_data)  # transfer to tensor step. Change to dataset

            if torch.cuda.is_available():
                real_data = real_data.to(device)
                input_img = input_img.to(device)

                try:
                    output = nn.MSELoss(input)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception

            # Train D & G
            d_error, g_error, loss_vgg, penalty, p_real, p_fake, pred_images = train_discriminator_generator(d_optimizer,
                                                                                                         g_optimizer,
                                                                                                         real_data,
                                                                                                         input_img,
                                                                                                         loss_function=vgg_loss,
                                                                                                         batch_size=batch_size)

            pred_images = generator(input_img)
            g_optimizer.zero_grad()
            if torch.cuda.is_available():
                nn.Module.to(device)          #write by myself
            loss_vgg = nn.MSELoss()(pred_images, real_data);
            loss_vgg.backward()
            g_optimizer.step()
            # Display Progress
            if torch.cuda.is_available():
                real_data = real_data.to(device)
                input_img = input_img.to(device)

            if (n_batch) % 120 == 0:
                display.clear_output(False)
                # Display Images
                psnr = PSNR(real_data, pred_images)
                ssim = SSIM(real_data, pred_images)
                update_loss_function(d_error, g_error, loss_vgg,
                                 penalty, p_real, p_fake, psnr, ssim, mode='train')
                d_error, g_error, loss_vgg_test, penalty, p_real, p_fake, pred_images = validate_discriminator_generator(real_test_img, input_test_img, batch_size=2, loss_function=vgg_loss)
                pred_images = generator(input_test_img)
                loss_vgg_test = nn.MSELoss()(real_test_img, pred_images)
                psnr = PSNR(real_test_img, pred_images)
                ssim = SSIM(real_test_img, pred_images)
                update_loss_function(d_error, g_error, loss_vgg_test,
                                 penalty, p_real, p_fake, psnr, ssim, mode='test')

                pred_images = generator(input_test_img.to(device)).data.to("cpu")
                display_status(epoch, num_epochs + 10, n_batch,
                           p_real, loss_vgg, p_fake, penalty,
                           batch_=int(len(data_loader.dataset) / batch_size))
                if epoch < 100:
                    epoch_ = '0' + str(epoch)
                else:
                    epoch_ = epoch
                plot_prediction(input_test_img, pred_images, real_test_img,
                            only_noise, plot=True,
                            epoch=str(epoch_) + "_" + str(n_batch))
                gc.collect()

# %%
#     fig, ax = plt.subplots(4, 2, figsize=(30, 20))
#     ax = ax.flatten()
#     i = 0
#     for key, values in loss_function.items():
#         #     if key in ['SSIM', 'loss_vgg', 'PSNR']:
#         ax[i].plot(values['test'][2:], label='test')
#         ax[i].plot(values['train'][2:], label='train')
#         ax[i].set_title(key)
#         ax[i].legend(loc='best')
#         i = i + 1

    # # %%
    # torch.save(generator.state_dict(), 'generator' + name)
    # torch.save(discriminator.state_dict(), 'discriminator' + name)
    #
    # np.save(name, loss_function)
    # fig.savefig('loss_gen' + name + '.png')

