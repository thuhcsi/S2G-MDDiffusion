import torch
from torchvision import models
import torch.nn as nn

def l1_loss_mask(inputs, targets, mask):
    loss = torch.sum(torch.abs(inputs - targets), dim=[1,2,3])
    xs = targets.size()
    ratio = torch.sum(mask, dim=[1,2,3]) * xs[1]  # mask: BWH1. if mask = BWH3, then remove '*3'
    loss_mean = torch.mean(loss / (ratio + 1e-12))  # avoid mask = 0
    return loss_mean

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value

def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv

def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value



class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, is_disc=None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                return self.loss(1 + prediction).mean()
            else:
                return (-prediction).mean()

        return loss
    
    
