import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """使用LN"""
    expansion = 4

    def __init__(self, in_planes, planes, kernel_size=7, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=in_planes,
            bias=False,
        )
        self.ln1 = LayerNorm(in_planes)
        self.conv2 = nn.Conv2d(
            in_planes,
            in_planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.conv3 = nn.Conv2d(
            in_planes * self.expansion,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x


        out = self.conv1(x)
        out = self.ln1(out)

        out = self.conv2(out)
        out = self.gelu(out)

        out = self.conv3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        # 将输入x添加回输出
        out += identity


        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers=[3, 3, 9, 3], kernel_size=7, in_planes=96, num_classes=100):
        super(ResNeXt, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=4, stride=4, bias=False)
        self.ln1 = LayerNorm(in_planes)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes * 8, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                LayerNorm(planes),
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                self.kernel_size,
                stride,
                downsample,
            )
        )
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: [N, C, H, W]
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return x
