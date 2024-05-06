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


class PipelineModel(torch.nn.Module):
    """
    模型汇总
    """

    def __init__(self, accalerator,
                 kp_extractor, dense_motion_network, inpainting_network, 
                 unet_generator, gan_discriminator, 
                 optimizer_unet, optimizer_gan,
                 scheduler_unet, scheduler_gan,
                 train_params, tps_checkpoint):
        super(PipelineModel, self).__init__()
        self.tps_checkpoint_path = tps_checkpoint
        self.accelerator = accalerator
        self.device = accalerator.device
        
        self.kp_extractor = kp_extractor.to(self.device)
        self.dense_motion_network = dense_motion_network.to(self.device)
        self.inpainting_network = inpainting_network.to(self.device)
        self.unet_generator = unet_generator.to(self.device)
        self.gan_discriminator = gan_discriminator.to(self.device)
        
        self.optimizer_unet = optimizer_unet
        self.optimizer_gan = optimizer_gan
        self.optimizers = [optimizer_unet, optimizer_gan]
        
        self.scheduler_unet = scheduler_unet
        self.scheduler_gan = scheduler_gan
        self.schedulers = [scheduler_unet, scheduler_gan]
        
        self.criterionGAN = GANLoss().to(self.device)
        self.lossNet = VGG16FeatureExtractor().to(self.device)

        self.train_params = train_params
        self.scales = train_params['scales']

        self.pyramid = ImagePyramide(self.scales, inpainting_network.num_channels).to(self.device)

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19().to(self.device)

    def setup(self):
        # 做好多卡准备
        self.kp_extractor, self.dense_motion_network, self.inpainting_network, \
            self.unet_generator, self.gan_discriminator,\
            self.optimizer, self.optimizer_unet, self.optimizer_gan,\
            self.scheduler_optimizer, self.scheduler_unet, self.scheduler_gan,\
            self.criterionGAN, self.lossNet, self.pyramid, self.vgg = self.accelerator.prepare(
                self.kp_extractor, self.dense_motion_network, self.inpainting_network,
                self.unet_generator, self.gan_discriminator,
                self.optimizer, self.optimizer_unet, self.optimizer_gan,
                self.scheduler_optimizer, self.scheduler_unet, self.scheduler_gan,
                self.criterionGAN, self.lossNet, self.pyramid, self.vgg
            )
        
        
        # load tps checkpoint
        unwrapped_kp_extractor = self.accelerator.unwrap_model(self.kp_extractor)
        unwrapped_dense_motion_network = self.accelerator.unwrap_model(self.dense_motion_network)
        unwrapped_inpainting_network = self.accelerator.unwrap_model(self.inpainting_network)
        if self.tps_checkpoint_path is not None:
            checkpoint = torch.load(self.tps_checkpoint_path)
            unwrapped_kp_extractor.load_state_dict(checkpoint['kp_detector'])
            unwrapped_dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
            unwrapped_inpainting_network.load_state_dict(checkpoint['inpainting_network'])
            
        # 设置好训练和验证模式
        self.kp_extractor.eval()
        self.dense_motion_network.eval()
        self.inpainting_network.eval()
        self.unet_generator.train()
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
    
    def optimize_parameters(self, x, epoch):
        # 前传
        self.forward(x, epoch)
        
        # 更新gan
        self.set_requires_grad(self.gan_discriminator, True) # 允许gan反传
        self.optimizer_gan.zero_grad()
        self.backward_gan() # 反传计算gan梯度
        self.optimizer_gan.step() 
        
        #更新生成器
        self.set_requires_grad(self.gan_discriminator, False) # 更新生成器，gan不计算梯度
        
        # 更新生成器 包括tps和unet
        self.optimizer.zero_grad()
        self.optimizer_unet.zero_grad()
        self.backward_generator()
        clip_grad_norm_(self.kp_extractor.parameters(), max_norm=10, norm_type = math.inf)
        clip_grad_norm_(self.dense_motion_network.parameters(), max_norm=10, norm_type = math.inf)
        self.optimizer.step()
        self.optimizer_unet.step()
        
        return self.loss_values, self.generated
        
    def forward(self, x, epoch):
        
        # tps阶段 loss不加权了 最后统一加权
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        if(epoch>=self.dropout_epoch):
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp 
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        
        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated = self.inpainting_network(x['source'], dense_motion)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        
        
        hand_mask = x['hand_mask'] # B,H,W
        hand_mask = hand_mask.unsqueeze(1) # B,1,H,W
        generated.update({'hand_mask': hand_mask})

        loss_values = {}
        
        # # box区域内的L1 loss
        # if self.loss_weights['hand'] != 0:
        #     # 所有手部的掩膜
        #     hand_mask = hand_mask.unsqueeze(1).repeat(1,3,1,1) # B,C,H,W
        #     num_pixels = hand_mask.sum()
        #     hand_loss = (torch.abs(x['driving'] - generated['prediction']) * hand_mask).sum() / num_pixels

        #     loss_values['hand_loss'] = self.loss_weights['hand'] * hand_loss
        

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        # reconstruction loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += value # 最后统一加权
            loss_values['perceptual'] = value_total

        # equivariance loss
        if self.loss_weights['equivariance_value'] != 0:
            transform_random = TPS(mode = 'random', bs = x['driving'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['driving'])
            transformed_frame = F.grid_sample(x['driving'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
        
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = kp_driving['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            loss_values['equivariance_value'] = value

        # warp loss：source经过enc后做warp和occlude和图像 v.s. driving经过encoder后只做occlude的图像(切断梯度)
        # 训练的是之前光流和encoder 对微调dec没影响
        if self.loss_weights['warp_loss'] != 0:
            occlusion_map = generated['occlusion_map']
            encode_map = self.inpainting_network.module.get_encode(x['driving'], occlusion_map) # 多卡，.module
            decode_map = generated['warped_encoder_maps']
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i]-decode_map[-i-1]).mean()

            loss_values['warp_loss'] = value
            
        
        # 结束tps阶段，generated['prediction']是tps后的结果
        merged_images1 = generated['prediction']  # 预测的局部模糊的图像 B,3,H,W
        masks = generated['occlusion_map'][-1] # 掩膜 B,1,H,W
        output_images2 = self.unet_generator(torch.cat((merged_images1, 1 - masks), dim=1)) # 生成的图像
        merged_images2 = merged_images1 * masks + output_images2 * (1 - masks) # mask的黑色部分用最新的，白色部分用旧的
        
        generated['unet_prediction'] = output_images2
        generated['merged_prediction'] = merged_images2
        
        self.x = x
        self.generated = generated
        self.loss_values = loss_values
        
    def backward_gan(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.gan_discriminator(self.generated['merged_prediction'].detach())
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
        mask = self.generated['occlusion_map'][-1] # B,1,H,W
        hand_mask = self.generated['hand_mask'] # B,1,H,W
        prediction = self.generated['prediction']
        unet_prediction = self.generated['unet_prediction']
        merged_prediction = self.generated['merged_prediction']
        # tps原本的损失
        loss_tps_valid = l1_loss_mask(prediction*mask, x['driving']*mask, mask)
        loss_tps_hole = l1_loss_mask(prediction*(1-mask), x['driving']*(1-mask), 1-mask)
        loss_tps_hand = l1_loss_mask(prediction*hand_mask, x['driving']*hand_mask, hand_mask) # hand_mask手部是1，其他是0
        loss_tps_l1 = loss_tps_valid + 3 * loss_tps_hole + 3 * loss_tps_hand # 按照比例设计的l1 loss
        loss_tps_content = self.loss_values['perceptual']
        loss_tps_equivariance = self.loss_values['equivariance_value']
        loss_tps_warp = self.loss_values['warp_loss']
        
        self.loss_values['tps_l1'] = loss_tps_l1
        
        # unet的损失
        loss_unet_valid = l1_loss_mask(unet_prediction*mask, x['driving']*mask, mask)
        loss_unet_hole = l1_loss_mask(unet_prediction*(1-mask), x['driving']*(1-mask), 1-mask)
        loss_unet_hand = l1_loss_mask(unet_prediction*hand_mask, x['driving']*hand_mask, hand_mask) # hand_mask手部是1，其他是0
        loss_unet_l1 = loss_unet_valid + 3 * loss_unet_hole + 3 * loss_unet_hand # 按照比例设计的l1 loss
            # 计算tv perc style等损失
        real_B_feats = self.lossNet(x['driving'])
        fake_B_feats = self.lossNet(unet_prediction)
        comp_B_feats = self.lossNet(merged_prediction)
        
        loss_unet_tv = TV_loss(merged_prediction*(1-mask))
        loss_unet_style = style_loss(real_B_feats, fake_B_feats) + style_loss(real_B_feats, comp_B_feats)
        loss_unet_content = perceptual_loss(real_B_feats, fake_B_feats) + perceptual_loss(real_B_feats, comp_B_feats)
        
        self.loss_values['unet_l1'] = loss_unet_l1
        self.loss_values['unet_tv'] = loss_unet_tv
        self.loss_values['unet_style'] = loss_unet_style
        self.loss_values['unet_content'] = loss_unet_content
        
        # gan的损失，merge图像需要迷惑gan  预测结果和1111算loss 所以是越小越好
        pred_fake = self.gan_discriminator(merged_prediction)
        loss_G_GAN = self.criterionGAN(pred_fake, True, is_disc=False)
        
        self.loss_values['gan_G'] = loss_G_GAN
        
        # 总损失
        loss = loss_tps_l1 + 0.05*loss_tps_content + 0.05*loss_tps_equivariance + 0.05*loss_tps_warp + \
            loss_unet_l1 + 0.05*loss_unet_content + 120*loss_unet_style + 0.1*loss_unet_tv\
                + 0.1*loss_G_GAN
        
        self.loss_values['total_loss'] = loss
                
        self.accelerator.backward(loss)
        
        
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
        
        
        
        
        