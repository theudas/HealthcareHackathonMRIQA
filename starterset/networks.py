import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet


class ClassicCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, batch_size=8):
        super(ClassicCNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.block1 = self.classic_cnn_block(input_channels, 32, 3, stride=2)
        self.block2 = self.classic_cnn_block(32, 64, 3, stride=2)
        self.maxpool1 = nn.MaxPool2d(2)

        self.block3 = self.classic_cnn_block(64, 128, 3)
        self.block4 = self.classic_cnn_block(128, 256, 3, stride=2)
        self.maxpool2 = nn.MaxPool2d(2)

        self.block5 = self.classic_cnn_block(256, 128, 3)
        self.block6 = self.classic_cnn_block(128, 64, 3, stride=2)
        self.block7 = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.maxpool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = torch.mean(x, dim=(2, 3))
        return x

    @staticmethod
    def classic_cnn_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


# add you own networks :)
import torch
import torch.nn.functional as F

def mish(x):
    return (x*torch.tanh(F.softplus(x)))

class mish_layer(nn.Module):
    def __init__(self):
        super(mish_layer, self).__init__()
    
    def forward(self, input):
        return mish(input)

class PhilsClassicCnn(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, batch_size=8):
        super(PhilsClassicCnn, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.block1 = self.classic_cnn_block(input_channels, 128, 7)
        self.maxpool1 = nn.MaxPool2d(2)

        self.block2 = self.classic_cnn_block(128, 128, 3)
        self.block3 = self.classic_cnn_block(128, 128, 3)

        self.block4 = self.classic_cnn_block(128, 128, 3)
        self.block5 = self.classic_cnn_block(128, 128, 3)

        self.block6 = self.classic_cnn_block(128, 128, 3)
        self.block7 = self.classic_cnn_block(128, 128, 3)

        self.block8 = self.classic_cnn_block(128, 128, 3)
        self.block9 = self.classic_cnn_block(128, 64, 3)
        self.block10 = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        res = self.maxpool1(x)

        x = self.block2(res)
        x = self.block3(x)
        x += res
        res = self.maxpool2(x)

        x = self.block4(res)
        x = self.block5(x)
        x += res
        res = self.maxpool3(x)

        x = self.block6(res)
        x = self.block7(x)
        x += res
        x = self.maxpool4(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = torch.mean(x, dim=(2, 3))
        return x

        
    def classic_cnn_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),

            nn.BatchNorm2d(out_channels),
            mish_layer()
        )

class CatNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, batch_size=8):
        super(CatNet, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        size = 82

        self.block1 = self.classic_cnn_block(input_channels, size, 7, 1, 3)
        self.maxpool1 = nn.MaxPool2d(2)

        self.block2 = skip_connection_block(size)
        self.block3 = skip_connection_block(size)
        self.block4 = skip_connection_block(size)
        self.block5 = skip_connection_block(size)

        self.block8 = skip_connection_block(size)
        self.block9 = skip_connection_block(size)
        self.block10 = nn.Conv2d(size, self.num_classes, 1)

    def forward(self, x):
        dropout_rate = 0.01

        x = self.block1(x)
        x = self.maxpool1(x)

        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        
        
        x = torch.sum(x, dim=(2, 3))
        x = torch.softmax(x, dim=-1)
        return x

    def classic_cnn_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),

            nn.BatchNorm2d(out_channels),
            mish_layer()
        )

class skip_connection_block(torch.nn.Module):
    def __init__(self, channels):
        super(skip_connection_block, self).__init__()
        self.conv_il = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.batn_il = nn.BatchNorm2d(channels)

        self.conv_ol = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.batn_ol = nn.BatchNorm2d(channels)

        self.conv_sl = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=2, padding=0)
        self.batn_sl = nn.BatchNorm2d(channels)

    def forward(self, x):
        il = self.conv_il(x)
        il = self.batn_il(il)
        il = mish_layer()(il)

        ol = self.conv_ol(il)
        ol = self.batn_ol(ol)

        sl = self.conv_sl(x)
        sl = self.batn_sl(sl)

        out = ol + sl
        out = mish_layer()(out)
        return out

class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pre = ClassicCNN.classic_cnn_block(1, 3)
        self.resnet = resnet(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, *inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = self.pre(x)
        x = self.resnet(x)
        return x