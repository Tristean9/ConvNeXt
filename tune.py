
import torch
import torch.nn as nn
from engine import train_model, evaluate_model
from model.convnext import ConvNeXt
from model import fewernorm
from utils.data_utils import load_data
from utils.log_utils import configure_logging


def tune():

    train_dir = "./dataset/MSR/train"
    train_loader = load_data(train_dir, batch_size=16)

    val_dir = "./dataset/MSR/val"
    val_loader = load_data(val_dir, batch_size=1)

    # 加载预训练模型
    model = fewernorm.ResNeXt(fewernorm.Bottleneck)
    weights_path = "trained_models/fewer_norms.pth"
    model.load_state_dict(torch.load(weights_path))

    # 替换分类头
    num_classes = 4  # 新任务的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model_name = "fewer_norms_tune_msr"

    logger = configure_logging(model_name)
    logger.info(f"{model_name} starts training !")
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        model_name=model_name,
        num_classes=num_classes,
        epochs=20,
        logger=logger,
    )


def test():
    test_dir = "./dataset/MSR/val"
    test_loader = load_data(test_dir, batch_size=1)

    # 加载预训练模型
    model = fewernorm.ResNeXt(fewernorm.Bottleneck)
    # 替换分类头
    num_classes = 4  # 新任务的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    weights_path = "trained_models/fewer_norms_tune_msr.pth"
    model.load_state_dict(torch.load(weights_path))

    model_name = "fewer_norms_tune_msr"

    logger = configure_logging(model_name)
    evaluate_model(data_loader=test_loader, model=model, logger=logger)


def main():
    tune()
    test()


if __name__ == "__main__":
    main()
