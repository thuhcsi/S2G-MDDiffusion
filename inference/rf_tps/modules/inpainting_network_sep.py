import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        up_blocks = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            decoder_in_feature = out_features * 2
            if i==num_down_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, dense_motion):
        out = self.first(source_image) 
        encoder_map = [out] # encoder保存的结果
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)
        # out是enc最后一层输出

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation']
        out_ij = self.deform_input(out.detach(), deformation) # 在encoder后detach
        out = self.deform_input(out, deformation)

        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map[0]) # out是enc最后一层特征warping

        warped_encoder_maps = []
        warped_encoder_maps.append(out_ij)

        for i in range(self.num_down_blocks):
            
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            out = self.up_blocks[i](out) # warping的特征经过一层decode
            
            encode_i = encoder_map[-(i+2)] # encoder倒二层特征
            encode_ij = self.deform_input(encode_i.detach(), deformation) # encoder倒二层特征warping
            encode_i = self.deform_input(encode_i, deformation) # 和out一起送入decoder第二层
            
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij = self.occlude_input(encode_ij, occlusion_map[occlusion_ind].detach())
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind]) # warping、occlusion后的该层输入
            warped_encoder_maps.append(encode_ij) # 这里存切断和encoder梯度的warping、occlusion特征

            if(i==self.num_down_blocks-1):
                break

            out = torch.cat([out, encode_i], 1) # 上一层输出和该层的warping、occlude拼接

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear',align_corners=True)

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2-i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map


class InpaintingNetworkEncoder(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True, **kwargs):
        super(InpaintingNetworkEncoder, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, dense_motion):
        
        out = self.first(source_image) 
        encoder_map = [out] # encoder保存的结果
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)
        # out是enc最后一层输出

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source'] ## dense_motion中的deformed_source根据多个tps得到的多张图

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation'] # 光流
        out_ij = self.deform_input(out.detach(), deformation) # 在encoder后detach
        out = self.deform_input(out, deformation)

        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map[0]) # out是enc最后一层特征warping

        
        warped_encoder_maps = []
        warped_encoder_maps.append(out_ij)

        for i in range(self.num_down_blocks):
            
            encode_i = encoder_map[-(i+2)] # encoder倒二层特征
            encode_ij = self.deform_input(encode_i.detach(), deformation) # encoder倒二层特征warping
            encode_i = self.deform_input(encode_i, deformation) # 和out一起送入decoder第二层
            
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij = self.occlude_input(encode_ij, occlusion_map[occlusion_ind].detach())
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind]) # warping、occlusion后的该层输入
            warped_encoder_maps.append(encode_ij) # 这里存切断和encoder梯度的warping、occlusion特征

            if(i==self.num_down_blocks-1):
                break

            # out = torch.cat([out, encode_i], 1) # 上一层输出和该层的warping、occlude拼接

        deformed_source = self.deform_input(source_image, deformation) # 这里的deformed_source是source根据合成光流做warping
        
        # 下面两行是decoder会用到的warp and occlude图，需要对两张source做拼接
        output_dict["deformed"] = deformed_source # source用完整光流做warping
        output_dict["warped_encoder_maps"] = warped_encoder_maps # 按照decoder输入的顺序存warp和occlude的结果

        occlusion_last = occlusion_map[-1]


        # 下面是decoder会用到的occlude图，需要对两张source做拼接
        output_dict["occlusion_last"] = occlusion_last

        return output_dict


class InpaintingNetworkDecoder(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, 
                 multi_mask = True, image_size = 256, **kwargs):
        super(InpaintingNetworkDecoder, self).__init__()
        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.image_size = image_size
        
        # warped_encoder_maps的weight
        weight_for_warpped_encoder_maps = []
        for i in range(self.num_down_blocks+1): # 0~3
            weight_for_warpped_encoder_maps.append(nn.Parameter(torch.full((2, self.image_size // (2 ** (3 - i)), self.image_size // (2 ** (3 - i))), 0.5)))
        self.weight_for_warpped_encoder_maps = nn.ParameterList(weight_for_warpped_encoder_maps)
        
        # deformed_source的weight
        self.weight_for_deformed_source = nn.Parameter(torch.full((2, self.image_size, self.image_size), 0.5))
        
        # occlusion_last的weight
        self.weight_for_occlusion_last = nn.Parameter(torch.full((2, self.image_size, self.image_size), 0.5))
        
        up_blocks = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            decoder_in_feature = out_features * 2
            if i==num_down_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, encoder_output_dict_1, encoder_output_dict_2): # 输入是两张source的参数字典
        
        warped_encoder_maps_1 = encoder_output_dict_1["warped_encoder_maps"]
        deformed_source_1 = encoder_output_dict_1["deformed"]
        occlusion_last_1 = encoder_output_dict_1["occlusion_last"]
        warped_encoder_maps_2 = encoder_output_dict_2["warped_encoder_maps"]
        deformed_source_2 = encoder_output_dict_2["deformed"]
        occlusion_last_2 = encoder_output_dict_2["occlusion_last"]
        # 两张source的warped_encoder_maps加权 bs,c,h,w
        warped_encoder_maps = []
        for i in range(self.num_down_blocks+1): # 0~3 i对应feature层数，下面[0][1]对应两张source的加权
            maps = warped_encoder_maps_1[i] * self.weight_for_warpped_encoder_maps[i][0] + warped_encoder_maps_2[i] * self.weight_for_warpped_encoder_maps[i][1]
            warped_encoder_maps.append(maps)
        
        # 两张source的deformed_source
        deformed_source = deformed_source_1 * self.weight_for_deformed_source[0] + deformed_source_2 * self.weight_for_deformed_source[1]
        
        # 两张source的occlusion_last
        occlusion_last = occlusion_last_1 * self.weight_for_occlusion_last[0] + occlusion_last_2 * self.weight_for_occlusion_last[1]
        
        output_dict = {}
        
        output_dict["deformed_source"] = deformed_source
        output_dict['occlusion_last'] = occlusion_last
        
        out = warped_encoder_maps[0]
        for i in range(self.num_down_blocks):
            
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            out = self.up_blocks[i](out) # warping的特征经过一层decode
            
            encode_i = warped_encoder_maps[i+1]
            
            if(i==self.num_down_blocks-1):
                break

            out = torch.cat([out, encode_i], 1) # 上一层输出和该层的warping、occlude拼接

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out
        
        output_dict

        return output_dict



