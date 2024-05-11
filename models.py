import torch.nn as nn
import torch.nn.functional as F
import torch

#18/34
class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, using_resnet=True):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.using_resnet = using_resnet

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.using_resnet:
            out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, using_resnet=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.using_resnet = using_resnet

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.using_resnet:
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, using_resnet=True, big_kernel=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        if big_kernel:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, int(self.in_channel), kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(int(self.in_channel), self.in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )


        self.layer1 = self._make_layer(block, 64, blocks_num[0], using_resnet=using_resnet)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, using_resnet=using_resnet)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, using_resnet=using_resnet)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, using_resnet=using_resnet)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, using_resnet=True):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, using_resnet=using_resnet))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, using_resnet=using_resnet))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

class cnn_classify(nn.Module):
    def __init__(self, ) -> None:
        super(cnn_classify, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.pooling = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 13 * 13, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x 

import torchvision

def freeze(resnet50):
    for param in resnet50.parameters():
        param.requires_grad = False

def changelayer(resnet50):
    # Modify the fully connected layer
    num_ftrs = resnet50.fc.in_features
    num_classes = 4

    # Replace the last fully connected layer with a new one for your custom classification task
    resnet50.fc = nn.Linear(num_ftrs, num_classes)

def get_model(model_type):
    nclass = 4
    if 'resnet18' in model_type:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=nclass, include_top=True, 
                      using_resnet=('without' not in model_type), big_kernel=('small' not in model_type))
    elif 'resnet34' in model_type:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=nclass, include_top=True, 
                      using_resnet=('without' not in model_type))
    elif 'resnet50' in model_type:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=nclass, include_top=True, 
                      using_resnet=('without' not in model_type))
    elif 'resnet101' in model_type:
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=nclass, include_top=True, 
                      using_resnet=('without' not in model_type))
    elif 'resnet152' in model_type:
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=nclass, include_top=True, 
                      using_resnet=('without' not in model_type))
    elif 'cnn' in model_type:
        return cnn_classify()
    elif 'Adamresnet' in model_type:
        Adamresnet = torchvision.models.resnet50(pretrained=True)
        freeze(Adamresnet)
        changelayer(Adamresnet)
        return Adamresnet