import torch
import torch.nn as nn

# CAM
class ChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 计算得到的注意力
        atten = self.sigmoid(avg_out + max_out)
        # 将输入矩阵乘以对应的注意力
        return x * atten

# SAM
class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        # 计算得到的注意力
        atten = self.conv1(atten)
        # 将输入矩阵乘以对应的注意力
        atten = self.sigmoid(atten)
        # 将输入矩阵乘以对应的注意力
        return x * atten

# CAM和SAM串行
class ChannelSpatialAttention2D(nn.Module):
    def __init__(self, in_planes) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.cam = ChannelAttention2D(in_planes)
        self.sam = SpatialAttention2D()
    def forward(self, x):
        x1 = self.sam(self.cam(x))
        return x1
    
if __name__ == '__main__':
    data = torch.rand(size=(1, 64, 512, 512))
    se = ChannelSpatialAttention2D(data.shape[1])
    op = se(data)
    print(op.shape)