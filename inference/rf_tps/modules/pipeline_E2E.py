from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, TPS
from torchvision import models
from torch.nn.utils import clip_grad_norm_
import numpy as np
from modules.loss import l1_loss_mask, VGG16FeatureExtractor, style_loss, perceptual_loss, TV_loss, GANLoss
import math

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}



class TPSModel(torch.nn.Module):
    def __init__(self, kp_extractor, dense_motion_network, inpainting_network, 
                 train_params, *kwargs):
        super(TPSModel, self).__init__()
        
        self.kp_extractor = kp_extractor
        self.dense_motion_network = dense_motion_network
        self.inpainting_network = inpainting_network
        
        self.train_params = train_params
        self.scales = train_params['scales']

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
            
    def forward(self, x):
        # tps阶段 loss不加权了 最后统一加权
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
        generated = self.inpainting_network(x['source'], dense_motion)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        
        
        hand_mask = x['hand_mask'] # B,H,W
        hand_mask = hand_mask.unsqueeze(1) # B,1,H,W
        generated.update({'hand_mask': hand_mask})
        
        return generated


class PipelineModel(torch.nn.Module):
    """
    模型汇总
    """

    def __init__(self, accelerator,
                 kp_extractor, dense_motion_network, inpainting_network, 
                 e2e_generator, gan_discriminator, 
                 optimizer_e2e, optimizer_gan,
                 scheduler_e2e, scheduler_gan,
                 train_params, tps_checkpoint):
        super(PipelineModel, self).__init__()
        self.tps_checkpoint_path = tps_checkpoint
        self.accelerator = accelerator
        self.device = accelerator.device
        self.epoch = 0
        self.loss_values = {}
        self.gan_start_epoch = train_params['gan_start_epoch']
        
        
        self.tps_model = TPSModel(kp_extractor, dense_motion_network, inpainting_network, train_params)
        # 下面三行是为了保存参数
        self.kp_extractor = kp_extractor
        self.dense_motion_network = dense_motion_network
        self.inpainting_network = inpainting_network
        
        self.tps_model.to(self.device)
        self.e2e_generator = e2e_generator.to(self.device)
        self.gan_discriminator = gan_discriminator.to(self.device)
        
        self.optimizer_e2e = optimizer_e2e
        self.optimizer_gan = optimizer_gan
        self.optimizers = [optimizer_e2e, optimizer_gan]
        
        self.scheduler_e2e = scheduler_e2e
        self.scheduler_gan = scheduler_gan
        self.schedulers = [scheduler_e2e, scheduler_gan]
        
        self.criterionGAN = GANLoss().to(self.device)
        self.lossNet = VGG16FeatureExtractor().to(self.device)


    def setup(self):
        # 做好多卡准备
        self.tps_model, \
            self.e2e_generator, self.gan_discriminator,\
            self.optimizer_e2e, self.optimizer_gan,\
            self.scheduler_e2e, self.scheduler_gan,\
            self.criterionGAN, self.lossNet = self.accelerator.prepare(
                self.tps_model,
                self.e2e_generator, self.gan_discriminator,
                self.optimizer_e2e, self.optimizer_gan,
                self.scheduler_e2e, self.scheduler_gan,
                self.criterionGAN, self.lossNet
            )
            
        # load tps checkpoint
        unwrapped_tps_model = self.accelerator.unwrap_model(self.tps_model)
        if self.tps_checkpoint_path is not None:
            checkpoint = torch.load(self.tps_checkpoint_path)
            unwrapped_tps_model.kp_extractor.load_state_dict(checkpoint['kp_detector'])
            unwrapped_tps_model.dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
            unwrapped_tps_model.inpainting_network.load_state_dict(checkpoint['inpainting_network'])
            
        self.tps_model.eval()
        self.e2e_generator.train()
        self.gan_discriminator.train()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def optimize_parameters(self, x):
        # 前传
        self.forward(x)
        
        # 更新gan
        self.set_requires_grad(self.gan_discriminator, True) # 允许gan反传
        self.optimizer_gan.zero_grad()
        self.backward_gan() # 反传计算gan梯度
        self.optimizer_gan.step() 
        
        #更新生成器
        self.set_requires_grad(self.gan_discriminator, False) # 更新生成器，gan不计算梯度
        
        # 更新生成器 包括tps和unet
        self.optimizer_e2e.zero_grad()
        self.backward_generator()
        self.optimizer_e2e.step()
        
        return self.loss_values, self.generated
        
    def forward(self, x):
        # tps固定 不算梯度
        with torch.no_grad():  
            generated = self.tps_model(x)
        # 结束tps阶段，generated['prediction']是tps后的结果，注意要detach
        merged_images1 = generated['prediction'].detach()  # 预测的局部模糊的图像 B,3,H,W
        masks = generated['occlusion_map'][-1].detach() # 掩膜 B,1,H,W
        output_images2 = self.e2e_generator(torch.cat((merged_images1, 1 - masks), dim=1)) # e2e生成
        merged_images2 = merged_images1 * masks + output_images2 * (1 - masks) # e2e混合，mask的黑色部分用最新的，白色部分用旧的

        generated['e2e_prediction'] = output_images2
        generated['e2e_merged_prediction'] = merged_images2

        
        self.x = x
        self.generated = generated

    def backward_gan(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.gan_discriminator(self.generated['e2e_merged_prediction'].detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_disc=True)
        # Real
        pred_real = self.gan_discriminator(self.x['driving'])
        self.loss_D_real = self.criterionGAN(pred_real, True, is_disc=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_values['gan_D'] = self.loss_D
        self.accelerator.backward(self.loss_D)
        
    def backward_generator(self):
        x = self.x
        mask = self.generated['occlusion_map'][-1].detach() # B,1,H,W
        prediction = self.generated['prediction'].detach() # B,3,H,W
        hand_mask = self.generated['hand_mask'] # B,1,H,W 在tps阶段做了unsqueeze 所以在generated里面
        e2e_prediction = self.generated['e2e_prediction']
        e2e_merged_prediction = self.generated['e2e_merged_prediction']
        
        # 计算一个tps输出的l1作为对比, 不用于参数更新！！！
        loss_tps_valid = l1_loss_mask(prediction*mask, x['driving']*mask, mask)
        loss_tps_hole = l1_loss_mask(prediction*(1-mask), x['driving']*(1-mask), 1-mask)
        loss_tps_hand = l1_loss_mask(prediction*hand_mask, x['driving']*hand_mask, hand_mask) # hand_mask手部是1，其他是0
        loss_tps_l1 = loss_tps_valid + 3 * loss_tps_hole + 5 * loss_tps_hand # 按照比例设计的l1 loss
        self.loss_values['tps_l1'] = loss_tps_l1

        # e2e的损失 
        loss_e2e_valid = l1_loss_mask(e2e_prediction*mask, x['driving']*mask, mask)
        loss_e2e_hole = l1_loss_mask(e2e_prediction*(1-mask), x['driving']*(1-mask), 1-mask)
        loss_e2e_hand = l1_loss_mask(e2e_prediction*hand_mask, x['driving']*hand_mask, hand_mask) # hand_mask手部是1，其他是0
        loss_e2e_l1 = loss_e2e_valid + 3 * loss_e2e_hole + 5 * loss_e2e_hand # 按照比例设计的l1 loss
            # 计算tv perc style等损失
        real_B_feats = self.lossNet(x['driving'])
        fake_B_feats_e2e = self.lossNet(e2e_prediction)
        comp_B_feats_e2e = self.lossNet(e2e_merged_prediction)
        
        loss_e2e_tv = TV_loss(e2e_merged_prediction*(1-mask))
        loss_e2e_style = style_loss(real_B_feats, fake_B_feats_e2e) + style_loss(real_B_feats, comp_B_feats_e2e)
        loss_e2e_content = perceptual_loss(real_B_feats, fake_B_feats_e2e) + perceptual_loss(real_B_feats, comp_B_feats_e2e)
        
        self.loss_values['e2e_l1'] = loss_e2e_l1
        self.loss_values['e2e_tv'] = loss_e2e_tv
        self.loss_values['e2e_style'] = loss_e2e_style
        self.loss_values['e2e_content'] = loss_e2e_content
        
        # gan的损失，merge图像需要迷惑gan  预测结果和1111算loss 所以是越小越好
        pred_fake = self.gan_discriminator(e2e_merged_prediction)
        loss_G_GAN = self.criterionGAN(pred_fake, True, is_disc=False)
        
        self.loss_values['gan_G'] = loss_G_GAN
        
        
        loss = loss_e2e_l1 + 0.1*loss_G_GAN + 0.05*loss_e2e_content + 120*loss_e2e_style + 0.1*loss_e2e_tv
        
        self.loss_values['total_loss'] = loss
                
        self.accelerator.backward(loss)
        
        
    def update_learning_rate(self):
        self.epoch += 1
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
        
        
        
        
        