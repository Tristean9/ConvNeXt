import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard日志记录器
import os
from utils.data_utils import mixup_data

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建具有标签平滑的真实标签分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
# Mixup 损失函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(train_loader, val_loader, model, model_name: str, logger, epochs=15):

    # 检查是否有可用的GPU，没有的话回退到CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用标签平滑技术
    criterion = LabelSmoothingLoss(classes=100, smoothing=0.1)
    # 使用AdamW优化器，并设置学习率和其他参数
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # 学习率调度器保持不变，这里使用StepLR作为示例
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 应用Mixup
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, alpha=1.0, use_cuda=torch.cuda.is_available()
            )
            images, labels_a, labels_b = map(
                torch.autograd.Variable, (images, labels_a, labels_b)
            )

            optimizer.zero_grad()
            outputs = model(images)

            # 计算Mixup损失
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        epoch_val_acc, epoch_val_loss = evaluate_model(val_loader, model, logger)

        logs_dir = os.path.join("./log", model_name)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # 使用tensorboard记录器
        writer = SummaryWriter(logs_dir)

        # 记录每个epoch的损失和准确率
        writer.add_scalar("training loss", epoch_loss, epoch)
        writer.add_scalar("training accuracy", epoch_acc, epoch)
        writer.add_scalar("training val accuracy", epoch_val_acc, epoch)
        writer.add_scalar("training val loss", epoch_val_loss, epoch)

        logger.info(
            f"Epoch {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )
        # 调整学习率
        scheduler.step()

    save_model_dir = os.path.join("./trained_models", model_name + ".pth")
    # 保存模型
    torch.save(model.state_dict(), save_model_dir)
    logger.info(f"Model saved with val accuracy: {100 * epoch_val_acc:.2f}%")
    logger.info("Finished Training")


def evaluate_model(data_loader, model, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # 在评估时不计算梯度
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    average_loss = total_loss / total
    accuracy = correct / total
    
    logger.info(f"Evaluating - Acc: {accuracy:.4f}, Loss: {average_loss:.4f}")

    return accuracy, average_loss
