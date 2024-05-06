import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, patch_size = 3, propagate_size = 3, stride = 1):
        super(SelfAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        
    def forward(self, foreground):
        bz, nc, w, h = foreground.size()
        background = foreground.clone()
        background = background
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i:i+1]
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim = True)**0.5
            conv_kernels = conv_kernels/norm_factor
            
            conv_result = F.conv2d(feature_map, conv_kernels, padding = self.patch_size//2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.avg_pool2d(conv_result, 3, 1, padding = 1)*9
            attention_scores = F.softmax(conv_result, dim = 1)
            
            feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride = 1, padding = self.patch_size//2)  # Note here, conv_kernels -> conv_kernels_all[i], keep amplitude information 
            final_output = feature_map
            output_tensor.append(final_output)
            att_score.append(attention_scores.permute(0,2,3,1).view(w*h,-1))  # 2D tensor, prob in dim=1

        return torch.cat(output_tensor, dim = 0), torch.cat(att_score, dim=0)
                
class AttentionModule(nn.Module):
    
    def __init__(self, inchannel, patch_size_list = [1], propagate_size_list = [3], stride_list = [1]):
        assert isinstance(patch_size_list, list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.att = SelfAttention(patch_size_list[0], propagate_size_list[0], stride_list[0])

        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size = 1)
        
    def forward(self, foreground):
        outputs, att_score = self.att(foreground)
        outputs = torch.cat([outputs, foreground],dim = 1)
        outputs = self.combiner(outputs)
        return outputs
