import os
import json
import logging
import torch
import torch.nn as nn

from engine import train_model, evaluate_model
from model.resnet import ResNet
from model.resnext import ResNeXt0, ResNeXt
from model.convnext import ConvNeXt
from model.bottleneck import Bottleneck, Bottleneckgroups, InvertedBottleneck, Moveup
from model import gelu, feweractivation, fewernorm, layernorm
from utils.data_utils import load_data
from utils.log_utils import configure_logging


def test():
    """
    函数“test”加载不同的模型，在测试数据集上评估它们的性能，并将结果保存在 JSON 文件中。
    """
    batch_size = 32

    test_dir = "./dataset/mini-imagenet/test"
    test_loader = load_data(test_dir, batch_size)

    model_list = [
        ResNet(Bottleneck, [3, 4, 6, 3]),  # 原始ResNet
        ResNet(Bottleneck, [3, 3, 9, 3]),  # 更改Bottleneck为[3, 3, 9, 3]
        ResNet(
            Bottleneck,
            [3, 3, 9, 3],
            conv1=nn.Conv2d(
                3, 64, kernel_size=4, stride=4, bias=False
            ),  # 更改干细胞卷积
        ),
        ResNeXt0(Bottleneckgroups, [3, 3, 9, 3]),  # 使用ResNeXt
        ResNeXt0(Bottleneckgroups, [3, 3, 9, 3], in_planes=96),  # 更改通道数为96
        ResNeXt(InvertedBottleneck, [3, 3, 9, 3], in_planes=96),
        ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=3),
        ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=5),
        ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=7),
        ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=9),
        gelu.ResNeXt(gelu.Bottleneck),
        feweractivation.ResNeXt(feweractivation.Bottleneck),
        fewernorm.ResNeXt(fewernorm.Bottleneck),
        layernorm.ResNeXt(layernorm.Bottleneck),
        ConvNeXt(),
    ]
    model_name_list = [
        "ResNet",
        "stage_ratio",
        "patchify_stem",
        "depth_conv",
        "width",
        "inverting_dims",
        "move",
        "kernel5",
        "kernel7",
        "kernel9",
        "gelu",
        "fewer_activations",
        "fewer_norms",
        "layernorm",
        "convnext",
    ]

    final_all_results = {}  # 储存所有模型的结果

    for model, model_name in zip(model_list, model_name_list):
        logger = configure_logging(model_name)
        # 加载权重
        weights_path = os.path.join("trained_models", model_name + ".pth")
        model.load_state_dict(torch.load(weights_path))
        test_acc, _ = evaluate_model(
            model=model, data_loader=test_loader, logger=logger
        )
        final_all_results[model_name] = test_acc
        logging.info(f"Evaluating - Acc: {test_acc:.2f}%")

    # 保存JSON数据到文件
    with open("log/final_all_results.json", "w") as file:
        file.write(json.dumps(final_all_results))


def train():
    """
    `train()` 函数使用指定的数据加载器为不同的模型定义训练过程，并记录训练进度。
    """

    train_dir = "./dataset/native-mini-imagenet/train"
    train_loader = load_data(train_dir, batch_size=16)

    val_dir = "./dataset/native-mini-imagenet/val"
    val_loader = load_data(val_dir, batch_size=1)

    model_list = [
        # ResNet(Bottleneck, [3, 4, 6, 3]),  # 原始ResNet
        # ResNet(Bottleneck, [3, 3, 9, 3]),  # 更改Bottleneck为[3, 3, 9, 3]
        # ResNet(
        #     Bottleneck,
        #     [3, 3, 9, 3],
        #     conv1=nn.Conv2d(
        #         3, 64, kernel_size=4, stride=4, bias=False
        #     ),  # 更改干细胞卷积
        # ),
        # ResNeXt0(Bottleneckgroups, [3, 3, 9, 3]),  # 使用ResNeXt
        # ResNeXt0(Bottleneckgroups, [3, 3, 9, 3], in_planes=96),  # 更改通道数为96
        # ResNeXt(InvertedBottleneck, [3, 3, 9, 3], in_planes=96),
        # ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=3),
        # ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=5),
        # ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=7),
        # ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=9),
        # ResNeXt(Moveup, [3, 3, 9, 3], in_planes=96, kernel_size=11),
        # gelu.ResNeXt(gelu.Bottleneck),
        # feweractivation.ResNeXt(feweractivation.Bottleneck),
        # fewernorm.ResNeXt(fewernorm.Bottleneck),
        # layernorm.ResNeXt(layernorm.Bottleneck),
        # ConvNeXt()
    ]
    model_name_list = [
        # "ResNet",
        # "stage_ratio",
        # "patchify_stem",
        # "depth_conv",
        # "width",
        # "inverting_dims",
        # "move",
        # "kernel5",
        # "kernel7",
        # "kernel9",
        # "kernel11",
        # "gelu",
        # "fewer_activations",
        # "fewer_norms",
        # "layernorm",
        # "convnext"
    ]

    for model, model_name in zip(model_list, model_name_list):
        if not os.path.exists(os.path.join("trained_models", model_name + ".pth")):
            logger = configure_logging(model_name)
            logger.info(f"{model_name} starts training !")
            train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                model_name=model_name,
                epochs=2,
                logger=logger,
            )
        else:
            logger = configure_logging(model_name)
            logger.info(f"{model_name} has trained !")


def main(self, arg):
    # train()
    test()


if __name__ == "__main__":
    main()
