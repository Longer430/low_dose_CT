'''
0. path of image
1. time bar
import time
import progressbar

p = progressbar.ProgressBar()
N = 1000

2.  # RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
img, noisy_img = img.cuda(), noisy_img.cuda()

3.
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()


'''
for i in p(range(N)):
    time.sleep(0.01)
# import numpy as np
#
# for i in range(10,20):
#     p = np.random.rand()
#     # p =1
#     print(p)
#
# #vgg_loss
#
#
# dim = 64                           # where?
# out_shape = 8                      # not exist
#
# Lambda = 10
# batch_size = 32
# num_epochs = 100


# ===================== test: show one figure    =====================

# t_2_start = time.perf_counter()                        #  count time
# target_path = os.path.join(dataset_LDCT, "ground_truth_validation", val_list[26])  #

# X_test = h5py.File(target_path, 'r')
# X_test_array =np.array(X_test['data'])
# X_test_tensor = torch.tensor(X_test_array, dtype=torch.float)
# print(type(X_test_tensor))
#
# # c = plt.imshow(X_test_tensor[70, :, :])
# # plt.show()
#
#
#
# # x, y = train[110]
# # x1, y1 = train[150]
# # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# # axes = axes.flatten()
# # axes[0].imshow(x[0], cmap="gray")
# # axes[0].axis('off')
# # axes[0].set_title('Noisy Image', fontsize=20)
# # # plt.subplot(122)
# # axes[1].imshow(y[0], cmap="gray")
# # axes[1].axis('off')
# # axes[1].set_title('Ground truth Image', fontsize=20)
# # # plt.subplot(221)
# # axes[2].imshow(x1[0], cmap="gray")
# # axes[2].axis('off')
# # # plt.subplot(222)
# # axes[3].imshow(y1[0], cmap="gray")
# # axes[3].axis('off')
# # plt.show()
# #
# #
# #
# # fig.savefig('sample', dpi = 100, bbox_inches='tight')
#
#
# t_2_end = time.perf_counter()                    #  count time
# print(t_2_start)
#

# # standardize it
        # #         img = (img - img.min()) /(img.max() - img.min())                   #?????????
        #
        #
        # # create noisy image
        # noise_type = self.img_dict[image_id]['noise']
        # if noise_type == "gaussian":
        #     noise = np.random.normal(0, 0.015, img.shape)
        # elif noise_type == "speckle":
        #     noise = img * np.random.randn(img.shape[0], img.shape[1]) / 5
        # noisy_img = img + noise
        # noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())
        # # if residual learning, ground-truth should be the noise
        # if self.only_noise: img = noise

        # # convert to PIL images
        # img = Image.fromarray(img)
        # noisy_img = Image.fromarray(noisy_img)
        #
        # # apply the same data augmentation transformations to the input and the ground-truth
        # p = np.random.rand()
        # if self.transforms and p < 0.5:
        #     self.t = T.Compose([T.RandomHorizontalFlip(1), T.ToTensor()])
        # else:
        #     self.t = T.Compose([T.ToTensor()])
        # img = self.t(img)
        # noisy_img = self.t(noisy_img)
        # return noisy_img, img