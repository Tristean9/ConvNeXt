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
    """
    函数“mixup_data”获取输入和输出数据，使用随机beta分布将它们混合，并返回混合输入数据以及原始和混合输出数据。
    
    :param x: 参数“x”通常是您想要混合的输入数据或特征。它可以是包含模型输入数据的张量或数组。
    :param y: `mixup_data` 函数中的 `y` 参数表示与输入数据 `x` 对应的目标标签或输出。它用于监督学习任务，其中模型经过训练以根据输入数据预测这些目标标签。
    :param alpha: `mixup_data` 函数中的 `alpha` 参数是一个超参数，用于控制输入和输出数据混合的强度。它用于计算混合比
    `lam`，该混合比决定每个数据点与另一个数据点的混合程度。
    :param use_cuda: `use_cuda` 参数是一个布尔标志，指示是否使用 CUDA 进行张量运算。
    :return: 函数“mixup_data”返回混合输入“mixed_x”、原始输出“y_a”、混洗后的输出“y_b”和混合系数“lam”。
    """
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
