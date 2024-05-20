import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandomErasing

def load_data(dataset_path, batch_size, shuffle=True):
    """
    加载数据集的函数。

    参数:
    - dataset_path: 数据集目录的路径。
    - batch_size: 批次大小。
    - shuffle: 是否打乱数据。
    """

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整图片大小
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 归一化
            RandomErasing(),   # 添加随机擦除数据增强
        ]
    )

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )

    print(
        f"Data from {dataset_path} has been loaded. Number of batches: {len(data_loader)}. Batch size: {batch_size}."
    )

    return data_loader


# Mixup 函数
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    ''' 返回混合后的输入输出对 '''
    if alpha <= 0:
        return x, y
    if use_cuda:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
    else:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
