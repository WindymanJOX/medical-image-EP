import torch
import torch.nn as nn

class VGG16_EP(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            # conv [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
            # [in_dim, 224, 224]
            # 0
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [64, 112, 112]
            # 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [128, 56, 56]
            # 10
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [256, 28, 28]
            # 17
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [512, 14, 14]
            # 24
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [512, 7, 7]
        )

        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, out_dim)
        )

        self.res = []

        for i in range(len(self.extractor)):
            self.extractor[i].register_forward_hook(self.get_activation())

    def getLayers(self):
        return self.extractor
    
    def get_activation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook

    def forward(self, x):
        self.res = []
        x = self.extractor(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = VGG16_EP(3, 2).cuda()
    data = torch.rand(4, 3, 224, 224).cuda()
    with torch.no_grad():
        _ = model(data)
    print()
    
