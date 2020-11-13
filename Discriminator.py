#%%

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3,
                stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3,
                stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3,
                stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear = nn.Sequential(
            torch.nn.Linear(256 * out_shape**2, 1024),
            nn.LeakyReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #print(x.shape)
        # Flatten and apply sigmoid
#         print(x.shape)
        x = x.view(-1, 256 * out_shape**2)
        x = self.linear(x)
        x = self.out(x)
        return x
