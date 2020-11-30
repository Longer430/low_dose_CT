from torch import nn

class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()
    """ FILL HERE """
    self.convlayers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )

  def forward(self,x):
    x = self.convlayers(x)
    return x


