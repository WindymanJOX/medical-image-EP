""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .se import ChannelSpatialAttention2D

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.Dropout(0.25, True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.Dropout(0.25, True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_channels, 64)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)

        factor = 2 if bilinear else 1

        self.conv5 = DoubleConv(512, 1024 // factor)
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)   # [64,512,512]

        x2 = self.maxpool(x1) # [64,256,256]
        x2 = self.conv2(x2)  # [128,256,256]

        x3 = self.maxpool(x2) # [128,128,128]
        x3 = self.conv3(x3)  # [256,128,128]

        x4 = self.maxpool(x3) # [256,64,64]
        x4 = self.conv4(x4)  # [512,64,64]

        x5 = self.maxpool(x4) # [512,32,32]
        x5 = self.conv5(x5)  # [1024,32,32]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetSE(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)
        self.csam1 = ChannelSpatialAttention2D(64)
        self.csam2 = ChannelSpatialAttention2D(128)
        self.csam3 = ChannelSpatialAttention2D(256)
        self.csam4 = ChannelSpatialAttention2D(512)
    
    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)   # [64,512,512]

        x2 = self.maxpool(x1) # [64,256,256]
        x2 = self.conv2(x2)  # [128,256,256]

        x3 = self.maxpool(x2) # [128,128,128]
        x3 = self.conv3(x3)  # [256,128,128]

        x4 = self.maxpool(x3) # [256,64,64]
        x4 = self.conv4(x4)  # [512,64,64]

        x5 = self.maxpool(x4) # [512,32,32]
        x5 = self.conv5(x5)  # [1024,32,32]

        x4 = self.csam4(x4)
        x = self.up1(x5, x4)

        x3 = self.csam3(x3)
        x = self.up2(x, x3)

        x2 = self.csam2(x2)
        x = self.up3(x, x2)

        x1 = self.csam1(x1)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

class UNetSEEP(UNetSE):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)

        self.res = []

        self.encoder_layers = [
            *self.conv1.double_conv, self.maxpool,
            *self.conv2.double_conv, self.maxpool,
            *self.conv3.double_conv, self.maxpool,
            *self.conv4.double_conv, self.maxpool,
            *self.conv5.double_conv
        ]

        mark = False
        for i in range(len(self.encoder_layers)):
            # maxpool挂过钩子就略过， 四个maxpool都是同一个对象
            if isinstance(self.encoder_layers[i], nn.MaxPool2d) and not mark:
                self.encoder_layers[i].register_forward_hook(self.get_activation())
                mark = True
            elif not isinstance(self.encoder_layers[i], nn.MaxPool2d):
                self.encoder_layers[i].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook

    def getLayers(self):
        # all model parts in order in one list 
        return self.encoder_layers
    
    def getFeatures(self):
        # all feature outputs in order in one list 
        return self.res

    def forward(self, x):
        self.res = []
        return super().forward(x)

class UNetEP(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)

        self.res = []

        self.encoder_layers = [
            *self.conv1.double_conv, self.maxpool,
            *self.conv2.double_conv, self.maxpool,
            *self.conv3.double_conv, self.maxpool,
            *self.conv4.double_conv, self.maxpool,
            *self.conv5.double_conv
        ]

        mark = False
        for i in range(len(self.encoder_layers)):
            # maxpool挂过钩子就略过， 四个maxpool都是同一个对象
            if isinstance(self.encoder_layers[i], nn.MaxPool2d) and not mark:
                self.encoder_layers[i].register_forward_hook(self.get_activation())
                mark = True
            elif not isinstance(self.encoder_layers[i], nn.MaxPool2d):
                self.encoder_layers[i].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook

    def getLayers(self):
        # all model parts in order in one list 
        return self.encoder_layers
    
    def getFeatures(self):
        # all feature outputs in order in one list 
        return self.res

    def forward(self, x):
        self.res = []
        return super().forward(x)

if __name__ == '__main__':
    import torch
    model = UNetEP(1, 1)
    data = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        op = model(data)
    print(op.shape)