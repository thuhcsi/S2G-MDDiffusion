import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from modules.attention import AttentionModule

class E2EGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, padding_type='reflect', use_dropout=False):
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
        super(E2EGenerator, self).__init__()
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
        model8 += [norm_layer(ngf * 8), ] # 2

        model9 = [nn.ReLU(True), ]
        model9 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model9 += [norm_layer(ngf * 8), ] # 4

        model10 = [nn.ReLU(True), ]
        model10 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model10 += [norm_layer(ngf * 8), ] # 8

        model11 = [nn.ReLU(True), ]
        model11 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model11 += [norm_layer(ngf * 8), ] # 16

        model12 = [nn.ReLU(True), ]
        model12 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model12 += [norm_layer(ngf * 4), ] # 32

        model13 = [nn.ReLU(True), ]
        model13 += [nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model13 += [norm_layer(ngf * 2), ] # 64

        model14 = [nn.ReLU(True), ]
        model14 += [nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model14 += [norm_layer(ngf), ] # 128size 128*2->64

        model15 = [nn.ReLU(True), ]
        model15 += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1), ]
        model15 += [nn.Tanh()] #256 64*2 -> 64

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

        # self.att_16 = AttentionModule(512)
        # self.att_32 = AttentionModule(256)
        # self.att_64 = AttentionModule(128)
        self.resblock_128_1 = ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resblock_128_2 = ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resblock_256_1 = ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resblock_256_2 = ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        final = [nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                      nn.Tanh()]
        self.final = nn.Sequential(*final) 

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
        # x_11 = self.att_16(x_11)
        x_12 = self.model12(torch.cat((x_4, x_11), 1))
        # x_12 = self.att_32(x_12)
        x_13 = self.model13(torch.cat((x_3, x_12), 1))
        # x_13 = self.att_64(x_13)
        x_14 = self.model14(torch.cat((x_2, x_13), 1))
        
        x_14 = self.resblock_128_1(x_14)
        x_14 = self.resblock_128_2(x_14)
        
        x_15 = self.model15(torch.cat((x_1, x_14), 1))
        
        x_15 = self.resblock_256_1(x_15)
        x_15 = self.resblock_256_2(x_15)
        
        output = self.final(x_15)

        return output
    
    
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, padding_type='reflect', downsampling=2):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = downsampling
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out