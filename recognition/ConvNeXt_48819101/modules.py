import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample: nn.Module=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.out_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(64, 64, 1)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.layer4 = self.make_layer(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def make_layer(self, inplanes, outplanes, stride):
        # Num blocks is always 2 as this is ResNet-18 not 34 or something
        layers = []
        downsample = None
        if inplanes==outplanes and stride == 1:
            layers.append(BasicBlock(inplanes, outplanes))
        else:
            downsample = nn.Sequential(nn.Conv2d(inplanes, outplanes, 1, stride), nn.BatchNorm2d(outplanes))
            layers.append(BasicBlock(inplanes, outplanes, stride, downsample))
        
        layers.append(BasicBlock(outplanes, outplanes))
        return nn.Sequential(*layers)