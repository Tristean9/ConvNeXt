import torch

from engine import train_model, evaluate_model
import sys

sys.path.append("model")
from resnet import ResNet, Bottleneck

sys.path.append("utils")
from data_utils import load_data


def train():
    batch_size = 32
    train_dir = "./dataset/mini-imagenet/train"
    train_loader = load_data(train_dir, batch_size)

    val_dir = "./dataset/mini-imagenet/val"
    val_loader = load_data(val_dir, batch_size)

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)
    train_model(train_loader=train_loader, val_loader=val_loader, model=model)


def test():
    batch_size = 32

    test_dir = "./dataset/mini-imagenet/test"
    test_loader = load_data(test_dir, batch_size)

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)
    # 加载权重
    weights_path = "./trained_models/best_mini_resnet50.pth"
    model.load_state_dict(torch.load(weights_path))

    test_acc, _ = evaluate_model(model=model, data_loader=test_loader)
    print(f"Evaluating - Acc: {test_acc:.2f}%")


def main():
    # train()
    test()


if __name__ == "__main__":
    main()
