import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class DB_FPN(nn.Module):
    """
    Feature Pyramid Network for DBNet. 
    It takes 4 scales from the backbone, upsamples them to the same 1/4 scale, 
    and concatenates them into a single dense feature map.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 1x1 Convs to unify channel dimensions
        self.conv2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.conv5 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        
        # 3x3 Smooth Convs that output 1/4 of the total channels (e.g., 64)
        # So when concatenated, they reconstruct the out_channels (256)
        self.smooth_conv5 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth_conv4 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth_conv3 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth_conv2 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)

    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Unify channels
        in5 = self.conv5(c5)
        in4 = self.conv4(c4)
        in3 = self.conv3(c3)
        in2 = self.conv2(c2)
        
        # Top-down pathways (Upsample and add)
        out4 = in4 + F.interpolate(in5, size=in4.shape[2:], mode='nearest')
        out3 = in3 + F.interpolate(out4, size=in3.shape[2:], mode='nearest')
        out2 = in2 + F.interpolate(out3, size=in2.shape[2:], mode='nearest')
        
        # Smooth and adjust channel depth
        p5 = self.smooth_conv5(in5)
        p4 = self.smooth_conv4(out4)
        p3 = self.smooth_conv3(out3)
        p2 = self.smooth_conv2(out2)
        
        # Upsample all to the scale of P2 (1/4 of original image size)
        p5 = F.interpolate(p5, size=p2.shape[2:], mode='nearest')
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='nearest')
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='nearest')
        
        # Concatenate along channel dimension -> (Batch, 256, H/4, W/4)
        return torch.cat([p5, p4, p3, p2], dim=1)


class DBHead(nn.Module):
    """
    The prediction head. DBNet uses two of these: one for the Probability Map 
    and one for the Threshold Map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        
        # Upsample 1/4 -> 1/2 (original image scale needs two x2 upsamplings)
        self.up_conv = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 2, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        
        # Upsample 1/2 -> 1/1 and output 1 channel (Single heatmap)
        self.conv2 = nn.ConvTranspose2d(in_channels // 4, 1, 2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


class DBNet(nn.Module):
    """
    Complete DBNet Architecture.
    """
    def __init__(self, k=50):
        super().__init__()
        self.k = k # Amplification factor for differentiable binarization
        
        # 1. Backbone: ResNet-18 (Lightweight and fast)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1) # 1/4 scale
        self.layer2 = resnet.layer2 # 1/8 scale
        self.layer3 = resnet.layer3 # 1/16 scale
        self.layer4 = resnet.layer4 # 1/32 scale
        
        # 2. FPN
        self.fpn = DB_FPN(in_channels_list=[64, 128, 256, 512], out_channels=256)
        
        # 3. Probability and Threshold Heads
        self.prob_head = DBHead(256)
        self.thresh_head = DBHead(256)

    def step_function(self, prob_map, thresh_map):
        """ The Differentiable Binarization Formula: 1 / (1 + e^(-k * (P - T))) """
        return torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))

    def forward(self, x):
        # Extract features
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN Fusion
        fpn_features = self.fpn([c2, c3, c4, c5])
        
        # Predict Maps
        prob_map = self.prob_head(fpn_features)
        
        # In inference, we skip the threshold map and binarization calculation entirely to save time.
        if not self.training:
            return prob_map
            
        thresh_map = self.thresh_head(fpn_features)
        
        # Calculate approximate binary map dynamically
        approx_binary_map = self.step_function(prob_map, thresh_map)
        
        return prob_map, thresh_map, approx_binary_map
    


