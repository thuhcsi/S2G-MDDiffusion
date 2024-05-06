import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from modules.attention import AttentionModule

class Unet256(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet256, self).__init__()
        # construct unet structure

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        model2 = [nn.LeakyReLU(0.2, True), ]
        model2 += [nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model2 += [norm_layer(ngf * 2), ]

        model3 = [nn.LeakyReLU(0.2, True), ]
        model3 += [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model3 += [norm_layer(ngf * 4), ]

        model4 = [nn.LeakyReLU(0.2, True), ]
        model4 += [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model4 += [norm_layer(ngf * 8), ]

        model5 = [nn.LeakyReLU(0.2, True), ]
        model5 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model5 += [norm_layer(ngf * 8), ]

        model6 = [nn.LeakyReLU(0.2, True), ]
        model6 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model6 += [norm_layer(ngf * 8), ]

        model7 = [nn.LeakyReLU(0.2, True), ]
        model7 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model7 += [norm_layer(ngf * 8), ]

        model8 = [nn.LeakyReLU(0.2, True), ]
        model8 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model8 += [norm_layer(ngf * 8), ]

        model9 = [nn.ReLU(True), ]
        model9 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model9 += [norm_layer(ngf * 8), ]

        model10 = [nn.ReLU(True), ]
        model10 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model10 += [norm_layer(ngf * 8), ]

        model11 = [nn.ReLU(True), ]
        model11 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model11 += [norm_layer(ngf * 8), ]

        model12 = [nn.ReLU(True), ]
        model12 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model12 += [norm_layer(ngf * 4), ]

        model13 = [nn.ReLU(True), ]
        model13 += [nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model13 += [norm_layer(ngf * 2), ]

        model14 = [nn.ReLU(True), ]
        model14 += [nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model14 += [norm_layer(ngf), ]

        model15 = [nn.ReLU(True), ]
        model15 += [nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), ]
        model15 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.model9 = nn.Sequential(*model9)
        self.model10 = nn.Sequential(*model10)
        self.model11 = nn.Sequential(*model11)
        self.model12 = nn.Sequential(*model12)
        self.model13 = nn.Sequential(*model13)
        self.model14 = nn.Sequential(*model14)
        self.model15 = nn.Sequential(*model15)

        self.att_16 = AttentionModule(512)
        self.att_32 = AttentionModule(256)
        self.att_64 = AttentionModule(128)

    def forward(self, input):
        """Standard forward"""
        x_1 = self.model1(input)
        x_2 = self.model2(x_1)
        x_3 = self.model3(x_2)
        x_4 = self.model4(x_3)
        x_5 = self.model5(x_4) 
        x_6 = self.model6(x_5)
        x_7 = self.model7(x_6)
        x_8 = self.model8(x_7)
        x_9 = self.model9(torch.cat((x_7, x_8), 1))
        x_10 = self.model10(torch.cat((x_6, x_9), 1))
        x_11 = self.model11(torch.cat((x_5, x_10), 1))
        x_11 = self.att_16(x_11)
        x_12 = self.model12(torch.cat((x_4, x_11), 1))
        x_12 = self.att_32(x_12)
        x_13 = self.model13(torch.cat((x_3, x_12), 1))
        x_13 = self.att_64(x_13)
        x_14 = self.model14(torch.cat((x_2, x_13), 1))
        output = self.model15(torch.cat((x_1, x_14), 1))

        return output