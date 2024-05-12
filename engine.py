import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard日志记录器
from tqdm import tqdm
import os

# 检查目录是否存在，如果不存在则创建
logs_dir = "./runs/resnet_training"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

writer = SummaryWriter(logs_dir)
# 使用tensorboard记录器


def train_model(train_loader, val_loader, model, epochs=30):

    # 检查是否有可用的GPU，没有的话回退到CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 学习率调度器，这里使用StepLR作为示例
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    high_accuracy = float("-inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        # 使用tqdm包装训练数据加载器，以显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            # 更新进度条的描述，为了简洁只显示当前轮次和损失
            pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        epoch_val_acc, epoch_val_loss = evaluate_model(val_loader, model)

        # 记录每个epoch的损失和准确率
        writer.add_scalar("training loss", epoch_loss, epoch)
        writer.add_scalar("training accuracy", epoch_acc, epoch)
        writer.add_scalar("training val accuracy", epoch_val_acc, epoch)
        writer.add_scalar("training val loss", epoch_val_loss, epoch)

        print(
            f"Epoch {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )
        # 调整学习率
        scheduler.step()

        if epoch_val_acc > high_accuracy:
            high_accuracy = epoch_val_acc
            # Save the model, but always overwrite with the highest accuracy model
            torch.save(model.state_dict(), "./trained_models/best_mini_resnet50.pth")
            print(f"Model saved at epoch {epoch+1} with accuracy: {100 * high_accuracy:.2f}%")

    print("Finished Training")


def evaluate_model(data_loader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式

    total_loss = 0.0
    correct = 0
    total = 0
    # 使用tqdm来显示测试进度
    pbar = tqdm(data_loader, desc="Evaluating")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # 在评估时不计算梯度
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 可以选择更新进度条描述来显示当前准确率
            pbar.set_description(f"Evaluating - Acc: {100. * correct / total:.2f}%")

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return accuracy, average_loss
