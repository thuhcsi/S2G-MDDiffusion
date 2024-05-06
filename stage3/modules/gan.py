from modules.spectral_norm import use_spectral_norm
import torch
import torch.nn as nn

class SNDiscriminator(nn.Module):
  def __init__(self, in_channels=3, use_sigmoid=False, use_sn=True, init_weights=True):
    super(SNDiscriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)

  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x