import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, n_channels, mid_channels,first_layer=False):
        super().__init__()

        if first_layer:
            self.conv1 = nn.Conv2d(n_channels, mid_channels, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv2d(n_channels, mid_channels, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
        conv = self.conv1(x)
        conv = self.relu1(x)
        conv = self.conv2(x)
        conv = self.relu2(x)

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
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        mid_channels = 64

        # downsampling
        k=1
        out = mid_channels*k
        inp = n_channels
        self.down1 = DoubleConv(inp,out,first_layer=True)
        self.pool1 = PostProcess()

        k=2
        out = mid_channels*k
        inp = out//2
        self.down2 = DoubleConv(inp,out)
        self.pool2 = PostProcess()

        k=4
        out = mid_channels*k
        inp = out//2
        self.down3 = DoubleConv(inp,out)
        self.pool3 = PostProcess()

        k=8
        out = mid_channels*k
        inp = out//2
        self.down4 = DoubleConv(inp,out)
        self.pool4 = PostProcess()
        
        # mid
        k=16
        out = mid_channels*k
        inp = out//2
        self.mid = DoubleConv(inp,out)

        # upsampling
        k=8
        out = mid_channels*k
        inp = inp*2
        self.trans4 = nn.ConvTranspose2d(inp, out, kernel_size=3, stride=2, padding=1)
        self.drop4 = nn.Dropout(p=0.5,inplace=True)
        self.deconv4 = DoubleConv(inp, out)

        k=4
        out = mid_channels*k
        inp = inp*2
        self.trans3 = nn.ConvTranspose2d(inp, out, kernel_size=3, stride=2, padding=1)
        self.drop3 = nn.Dropout(p=0.5,inplace=True)
        self.deconv3 = DoubleConv(inp, out)

        k=2
        out = mid_channels*k
        inp = inp*2
        self.trans2 = nn.ConvTranspose2d(inp, out, kernel_size=3, stride=2, padding=1)
        self.drop2 = nn.Dropout(p=0.5,inplace=True)
        self.deconv2 = DoubleConv(inp, out)

        k=1
        out = mid_channels*k
        inp = inp*2
        self.trans1 = nn.ConvTranspose2d(inp, out, kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout(p=0.5,inplace=True)
        self.deconv1 = DoubleConv(inp, out)

        # output layer
        self.out = nn.Conv2d(out, n_classes, kernel_size=1, padding=1)

    def forward(self, x):
        # downsampling
        conv1 = self.down1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.down4(pool3)
        pool4 = self.pool4(conv4)

        # mid
        conv_mid = self.mid(pool4)

        # upsampling

        deconv4 = self.trans4(conv_mid)
        uconv4 = torch.cat([deconv4, conv4], dim=1)
        uconv4 = self.drop4(uconv4)
        uconv4 = self.deconv4(uconv4)

        deconv3 = self.trans3(uconv4)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.drop3(uconv3)
        uconv3 = self.deconv3(uconv3)

        deconv2 = self.trans2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.drop2(uconv2)
        uconv2 = self.deconv2(uconv2)

        deconv1 = self.trans1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.drop1(uconv1)
        uconv1 = self.deconv1(uconv1)

        # output layer
        out = self.out(uconv1)

        return out

pixels = 32*32
model = UNet(n_channels=pixels,n_classes=9)
print(model)