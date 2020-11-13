

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        x = x*255.0
        x = x.repeat(1, 3, 1, 1)
        x[:, 0,:,:]-=103.939
        x[:, 1,:,:]-=116.779
        x[:, 2,:,:]-=123.68
        out = self.feature_extractor(x)
        return out
features_extractor = FeatureExtractor()
for param in features_extractor.parameters():
    param.requires_grad = False

