import torch
from engine import train_model, evaluate_model
from model.resnet import ResNet, Bottleneck
from utils.data_utils import load_data
from utils.log_utils import configure_logging

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
    
    
    train_dir = "./dataset/native-mini-imagenet/train"
    train_loader = load_data(train_dir, batch_size = 32)

    val_dir = "./dataset/native-mini-imagenet/val"
    val_loader = load_data(val_dir, batch_size=1)
    
    model_list = [ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)]
    model_name_list = ["ResNet"]
    
    for model, model_name in zip(model_list, model_name_list):
        configure_logging(model_name)
        train_model(train_loader=train_loader, val_loader=val_loader, model=model, model_name=model_name)
        
    

if __name__ == "__main__":
    main()
