import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, input, output,first_layer=False):
        super().__init__()

        self.conv1 = nn.Conv2d(input, output, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.conv1(x)
        conv = self.relu(conv)
        conv = self.conv2(conv)
        conv = self.relu(conv)
        return conv

class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5,inplace=True)

    def forward(self, x):
        pool = self.pool(x)
        pool = self.drop(pool)
        return pool

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, mid_channels = 64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # self.pool = nn.MaxPool2d(2)
        self.pool = PostProcess()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.mid = DoubleConv(512,1024)

        self.up4 = DoubleConv(1024 + 512, 512)
        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)

        self.out = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        # downsampling
        conv1 = self.down1(x)
        pool1 = self.pool(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.down3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.down4(pool3)
        pool4 = self.pool(conv4)
        
        # mid
        mid = self.mid(pool4)

        # upsampling
        deconv4 = self.upsample(mid)
        deconv4 = torch.cat([deconv4,conv4],dim=1)
        deconv4 = self.up4(deconv4)
        
        deconv3 = self.upsample(deconv4)
        deconv3 = torch.cat([deconv3,conv3],dim=1)
        deconv3 = self.up3(deconv3)

        deconv2 = self.upsample(deconv3)
        deconv2 = torch.cat([deconv2,conv2],dim=1)
        deconv2 = self.up2(deconv2)

        deconv1 = self.upsample(deconv2)
        deconv1 = torch.cat([deconv1,conv1],dim=1)
        deconv1 = self.up1(deconv1)

        # output layer
        out = self.out(deconv1)

        return out

# pixels = 32*32
# model = UNet(n_channels=pixels,n_classes=9)
# print(model)
