import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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



