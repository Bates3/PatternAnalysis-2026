import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample: nn.Module=None):
        super().__init__()
        expansion_factor = 4

        self.conv1 = nn.Conv2d(inplanes, outplanes, 1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(outplanes, outplanes*expansion_factor, 1)
        self.bn3 = nn.BatchNorm2d(outplanes*expansion_factor)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.expansion_factor = 4
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 3, 1)
        self.layer2 = self.make_layer(128, 4, 2)
        self.layer3 = self.make_layer(256, 6, 2)
        self.layer4 = self.make_layer(512, 3, 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512 * self.expansion_factor, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def make_layer(self, planes, num_blocks, stride):
        
        outplanes = planes*self.expansion_factor
     
        downsample = None

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, outplanes, 1, stride), nn.BatchNorm2d(outplanes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = outplanes
        for i in range(1, num_blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)