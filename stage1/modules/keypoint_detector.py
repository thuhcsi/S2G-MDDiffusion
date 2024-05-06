from torch import nn
import torch
from torchvision import models

class KPDetector(nn.Module):
    """
    Predict keypoints.
    """

    def __init__(self, num_tps, num_kp_per_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps
        self.num_kp_per_tps = num_kp_per_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps*num_kp_per_tps*2)

        
    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape # b,20*5*2
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1   # normalize to [-1,1]
        out = {'fg_kp': fg_kp.view(bs, self.num_tps*self.num_kp_per_tps, -1)} # b,20*5,2

        return out
