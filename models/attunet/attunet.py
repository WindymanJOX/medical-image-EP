from torch import nn
import torch

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.Dropout(0.25, True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

class AttUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(AttUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)   # [64,512,512], 0, 1

        x2 = self.Maxpool(x1) # [64,256,256]
        x2 = self.Conv2(x2)  # [128,256,256], 2, 3

        x3 = self.Maxpool(x2) # [128,128,128]
        x3 = self.Conv3(x3)  # [256,128,128], 4, 5

        x4 = self.Maxpool(x3) # [256,64,64]
        x4 = self.Conv4(x4)  # [512,64,64], 6, 7

        x5 = self.Maxpool(x4) # [512,32,32]
        x5 = self.Conv5(x5)  # [1024,32,32], 8, 9

        # decoding + concat path
        d5 = self.Up5(x5)    # [512,64,64]
        x4 = self.Att5(g=d5, x=x4)  # [512,64,64],[512,64,64], 7
        d5 = torch.cat((x4, d5), dim=1)  # [1024,64,64]
        d5 = self.Up_conv5(d5)  # [512,64,64]

        d4 = self.Up4(d5)    # [256,128,128]
        x3 = self.Att4(g=d4, x=x3)  # [256,128,128],[256,128,128], 5
        d4 = torch.cat((x3, d4), dim=1)  # [512,128,128]
        d4 = self.Up_conv4(d4)  # [256,128,128]

        d3 = self.Up3(d4)    # [128,256,256]
        x2 = self.Att3(g=d3, x=x2)  # [128,256,256],[128,256,256], 3
        d3 = torch.cat((x2, d3), dim=1)  # [256,256,256]
        d3 = self.Up_conv3(d3)  # [128,256,256]

        d2 = self.Up2(d3)    # [64,512,512]
        x1 = self.Att2(g=d2, x=x1)  # [64,512,512],[64,512,512], 1
        d2 = torch.cat((x1, d2), dim=1)  # [128,512,512]
        d2 = self.Up_conv2(d2)  # [64,512,512]

        d1 = self.Conv_1x1(d2)  # [2,512,512]

        return d1
    
class AttUNetEPWA(AttUNet):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__(n_channels, n_classes)

        self.encoder_layers = [
            *self.Conv1.conv, self.Maxpool,
            *self.Conv2.conv, self.Maxpool,
            *self.Conv3.conv, self.Maxpool,
            *self.Conv4.conv, self.Maxpool,
            *self.Conv5.conv
        ]

        self.att_layers = [
            self.Att5, self.Att4, self.Att3, self.Att2
        ]

        self.res = []
        self.res_att = []
        
        mark = False
        for i in range(len(self.encoder_layers)):
            # maxpool挂过钩子就略过， 四个maxpool都是同一个对象
            if isinstance(self.encoder_layers[i], nn.MaxPool2d) and not mark:
                self.encoder_layers[i].register_forward_hook(self.getActivation())
                mark = True
            elif not isinstance(self.encoder_layers[i], nn.MaxPool2d):
                self.encoder_layers[i].register_forward_hook(self.getActivation())
        
        for i in range(len(self.att_layers)):
            self.att_layers[i].register_forward_hook(self.getHookAtt())
            
    def getActivation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook
    
    def getHookAtt(self):
        def hookAtt(model, input, output):
            self.res_att.append(output.squeeze(0).cpu().numpy())
        return hookAtt
    
    def getLayers(self):
        return self.encoder_layers
    
    def getFeatures(self):
        # 将特定层的卷积输出换成经过注意力模块后的输出 
        j = 3
        for i in range(3, len(self.encoder_layers), 7):
            assert self.res[i].shape == self.res_att[j].shape
            self.res[i] = self.res_att[j]
            j -= 1
            if j<0:
                break
        return self.res
    
    def forward(self, x):
        self.res = []
        self.res_att = []
        return super().forward(x)

class AttUNet_EP(AttUNet):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__(n_channels, n_classes)

        self.encoder_layers = [
            *self.Conv1.conv, self.Maxpool,
            *self.Conv2.conv, self.Maxpool,
            *self.Conv3.conv, self.Maxpool,
            *self.Conv4.conv, self.Maxpool,
            *self.Conv5.conv
        ]

        self.res = []

        mark = False
        for i in range(len(self.encoder_layers)):
            # maxpool挂过钩子就略过， 四个maxpool都是同一个对象
            if isinstance(self.encoder_layers[i], nn.MaxPool2d) and not mark:
                self.encoder_layers[i].register_forward_hook(self.getActivation())
                mark = True
            elif not isinstance(self.encoder_layers[i], nn.MaxPool2d):
                self.encoder_layers[i].register_forward_hook(self.getActivation())
            

    def getActivation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook
    
    def getLayers(self):
        return self.encoder_layers
    
    def getFeatures(self):
        return self.res
    
    def forward(self, x):
        self.res = []
        return super().forward(x)

if __name__ == '__main__':
    net = AttUNetEPWA(1, 2)
    data = torch.rand(1, 1, 512, 512)
    with torch.no_grad():
        op = net(data)
    res = net.getFeatures()
    print(op.shape)