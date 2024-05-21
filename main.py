import os
import logging
import torch
import torch.nn as nn
from engine import train_model, evaluate_model
from model import resnet
from model import resnext
from utils.data_utils import load_data
from utils.log_utils import configure_logging


def test():
    batch_size = 32

    test_dir = "./dataset/mini-imagenet/test"
    test_loader = load_data(test_dir, batch_size)

    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # 加载权重
    weights_path = "./trained_models/best_mini_resnet50.pth"
    model.load_state_dict(torch.load(weights_path))

    test_acc, _ = evaluate_model(model=model, data_loader=test_loader)
    print(f"Evaluating - Acc: {test_acc:.2f}%")


def main():

    train_dir = "./dataset/native-mini-imagenet/train"
    train_loader = load_data(train_dir, batch_size=32)

    val_dir = "./dataset/native-mini-imagenet/val"
    val_loader = load_data(val_dir, batch_size=1)

    model_list = [
        resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3]),  # 原始ResNet
        resnet.ResNet(resnet.Bottleneck, [3, 3, 9, 3]),  # 更改Bottleneck为[3, 3, 9, 3]
        resnet.ResNet(
            resnet.Bottleneck,
            [3, 3, 9, 3],
            conv1=nn.Conv2d(
                3, 64, kernel_size=4, stride=4, bias=False
            ),  # 更改干细胞卷积
        ),
        resnext.ResNeXt(resnext.Bottleneck, [3, 3, 9, 3]),  # 使用ResNeXt
        resnext.ResNeXt(
            resnext.Bottleneck, [3, 3, 9, 3], in_planes=96  # 更改通道数为96
        ),
    ]
    model_name_list = ["ResNet", "stage_ratio", "patchify_stem", "depth_conv", "width"]

    for model, model_name in zip(model_list, model_name_list):
        # logging.info(os.path.join("trained_models", model_name + ".pth"))
        if not os.path.exists(os.path.join("trained_models", model_name + ".pth")):
            logger = configure_logging(model_name)
            logger.info(f"{model_name} starts training !")
            train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                model_name=model_name,
                epochs=3,
                logger=logger
            )
        else:
            logger = configure_logging(model_name)
            logger.info(f"{model_name} has trained !")


if __name__ == "__main__":
    main()
