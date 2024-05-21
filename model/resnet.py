import torch
import torch.nn as nn


class ResNet(nn.Module):
    """docstring for ResNet."""

    def __init__(
        self,
        block,
        layers,
        conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        num_classes=100,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 初始的干细胞卷积
        self.conv1 = conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 层1-4使用_make_layer函数构建
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，输出为num_classes个类别
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()  # 调用初始化函数

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
        # 如果步长不为1或者输入输出的维度不一致，需要使用downsample进行下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 添加第一个bottleneck块，可能带有downsample
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # 添加剩余的bottleneck块
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过4个层级
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 平均池化层和全连接层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平所有维度，除了批次
        x = self.fc(x)

        return x
