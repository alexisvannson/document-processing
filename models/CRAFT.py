import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class DoubleConv(nn.Module):
    """Standard U-Net convolutional block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load Pretrained VGG16 with Batch Norm
        vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        
        # 2. Slice VGG16 into blocks for U-Net skip connections
        self.slice1 = torch.nn.Sequential(*vgg[0:12])   # -> output shape (N, 128, H/2, W/2)
        self.slice2 = torch.nn.Sequential(*vgg[12:22])  # -> output shape (N, 256, H/4, W/4)
        self.slice3 = torch.nn.Sequential(*vgg[22:32])  # -> output shape (N, 512, H/8, W/8)
        self.slice4 = torch.nn.Sequential(*vgg[32:42])  # -> output shape (N, 512, H/16, W/16)
        
        # We need an extra block to simulate VGG's layer 5 pooling
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        # 3. U-Net style Decoder (Upsampling and Concatenating)
        self.upconv1 = DoubleConv(1024 + 512, 256, 256)
        self.upconv2 = DoubleConv(256 + 512, 128, 128)
        self.upconv3 = DoubleConv(128 + 256, 64, 64)
        self.upconv4 = DoubleConv(64 + 128, 32, 32)
        
        # 4. Final Output Head: 2 Channels (Region Map and Affinity Map)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1) # 2 Output Channels!
        )

    def forward(self, x):
        # Forward pass through VGG blocks (Encoding)
        c1 = self.slice1(x)
        c2 = self.slice2(c1)
        c3 = self.slice3(c2)
        c4 = self.slice4(c3)
        c5 = self.slice5(c4)
        
        # U-Net Decoding (Upsample -> Concat -> Convolve)
        x = F.interpolate(c5, size=c4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c4], dim=1)
        x = self.upconv1(x)
        
        x = F.interpolate(x, size=c3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.upconv2(x)
        
        x = F.interpolate(x, size=c2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.upconv3(x)
        
        x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.upconv4(x)
        
        # Generate Region and Affinity maps
        out = self.conv_cls(x)
        return out