import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, survival_prob=0.8):
        super(Bottleneck, self).__init__()
        # 1*1 卷积，降低维度
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 3*3 卷积，处理特征
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 1*1 卷积，扩展维度
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.survival_prob = survival_prob
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 在这里实现随机深度
        if self.training and torch.rand(1) < self.survival_prob:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        elif not self.training:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要进行下采样，则应用downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将输入x添加回输出
        out += identity
        out = self.relu(out)

        return out


class Bottleneckgroups(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneckgroups, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 使用分组卷积
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        # 将输入x添加回输出
        out += identity
        out = self.relu(out)

        return out


class InvertedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, downsample=None):
        super(InvertedBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * self.expansion)

        self.conv2 = nn.Conv2d(
            planes * self.expansion,
            planes * self.expansion,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.conv3 = nn.Conv2d(
            planes * self.expansion,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        # 将输入x添加回输出
        out += identity
        out = self.relu(out)

        return out


class Moveup(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, kernel_size=7, stride=1, downsample=None):
        super(Moveup, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=in_planes,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(
            in_planes,
            in_planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            in_planes * self.expansion,
        )

        self.conv3 = nn.Conv2d(
            in_planes * self.expansion,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print("x",x.shape)

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
        #     print("identity", identity.shape)
        # print("out", out.shape)
        # 将输入x添加回输出
        out += identity
        out = self.relu(out)

        return out
