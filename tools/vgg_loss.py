 ============================ step 3/5 loss function ============================
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