"""
MNIST 手写数字识别（入门学习版）

整体流程导读：
1) 数据准备：使用 torchvision.datasets.MNIST + transforms 做预处理
2) 模型定义：SimpleNN（多层感知机 MLP）把 28x28 图像映射到 10 个类别 logits
3) 训练循环：zero_grad -> forward -> loss -> backward -> step
4) 评估：关闭梯度，仅做前向推理，统计测试集 loss/accuracy
5) 可视化：展示部分预测结果，直观看模型是否学会识别数字
"""

import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """
    选择训练设备。

    输入：无
    输出：torch.device（mps/cuda/cpu）
    作用：尽可能使用硬件加速。
    为什么这样做：同一份代码可在 Mac(MPS)、NVIDIA(CUDA)、CPU 之间切换。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(batch_size: int = 128):
    """
    构建训练/测试 DataLoader。

    输入：
    - batch_size: 每个 batch 的样本数

    输出：
    - train_loader: 训练集 DataLoader
    - test_loader: 测试集 DataLoader

    作用：把原始图片整理成可迭代的小批量 Tensor。
    为什么这样做：mini-batch 训练更稳定、更高效。
    """
    # transforms：把 PIL 图片转成 Tensor，并标准化到更适合神经网络训练的分布
    # ToTensor() 后图像形状：[1, 28, 28]，像素范围约 [0, 1]
    # Normalize(mean, std) 后：x' = (x - mean) / std
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # shuffle=True：训练时打乱样本顺序，降低批次偏差
    # shuffle=False：测试时通常不需要打乱
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SimpleNN(nn.Module):
    """
    一个简单的全连接神经网络（MLP）：28x28 -> 10 类数字。

    输入张量形状（单 batch）：[B, 1, 28, 28]
    输出张量形状：             [B, 10]
    输出是 logits（未经过 softmax 的分数）。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),             # [B,1,28,28] -> [B,784]
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),       # 10 个类别（数字 0~9）
        )

    def forward(self, x):
        """
        前向传播：输入图像，输出 logits。
        """
        return self.net(x)


def train_one_epoch(model, loader, device, criterion, optimizer):
    """
    训练一个 epoch。

    输入：
    - model: 神经网络模型
    - loader: 训练集 DataLoader
    - device: 训练设备
    - criterion: 损失函数（这里是 CrossEntropyLoss）
    - optimizer: 优化器（这里是 Adam）

    输出：
    - avg_loss: 本 epoch 平均训练损失
    - acc: 本 epoch 训练准确率（%）

    为什么要分函数：把“训练逻辑”封装起来，主流程更清晰。
    """
    model.train()  # 进入训练模式（启用如 Dropout/BN 的训练行为；本例虽未用，但习惯保留）
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # images shape: [B, 1, 28, 28]
        # labels shape: [B]
        images = images.to(device)
        labels = labels.to(device)

        # ===== 训练闭环 5 步 =====
        # 1) 清空梯度：防止与上一个 batch 的梯度累加
        optimizer.zero_grad()

        # 2) 前向传播：得到 logits，形状 [B, 10]
        logits = model(images)

        # 3) 计算损失：CrossEntropyLoss 内部包含 log-softmax + NLL
        loss = criterion(logits, labels)

        # 4) 反向传播：根据损失计算每个参数的梯度
        loss.backward()

        # 5) 参数更新：按优化器规则更新参数
        optimizer.step()
        # ========================

        # 累计统计：用于计算 epoch 平均 loss 和 accuracy
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)  # 取每个样本得分最高的类别作为预测
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    在测试集上评估模型。

    输入：与训练函数类似，但不需要 optimizer。
    输出：平均测试损失和测试准确率。

    为什么用 @torch.no_grad()：
    - 不构建计算图，减少显存/内存占用
    - 推理更快
    - 评估阶段不需要更新参数
    """
    model.eval()  # 进入评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


@torch.no_grad()
def show_predictions(model, loader, device, num_images: int = 8):
    """
    可视化部分预测结果。

    输入：
    - model/loader/device: 与评估类似
    - num_images: 展示多少张图

    输出：无（弹出图像窗口）

    说明：
    - P 表示 Predicted（模型预测）
    - T 表示 True（真实标签）
    """
    model.eval()
    dataset = loader.dataset
    num_images = min(num_images, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_images)

    images = torch.stack([dataset[idx][0] for idx in indices]).to(device)
    labels = torch.tensor([dataset[idx][1] for idx in indices])

    logits = model(images)
    preds = logits.argmax(dim=1).cpu()
    images = images.cpu()

    plt.figure(figsize=(14, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)

        # 反标准化：把 x' 还原到接近原像素分布，便于人眼观察
        # x = x' * std + mean
        img = images[i].squeeze() * 0.3081 + 0.1307
        plt.imshow(img, cmap="gray")
        plt.title(f"P:{preds[i].item()} / T:{labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    主流程入口。

    学习时建议阅读顺序：
    main -> build_dataloaders -> SimpleNN -> train_one_epoch -> evaluate -> show_predictions
    """
    torch.manual_seed(42)  # 固定随机种子，便于复现实验
    device = get_device()
    print(f"Using device: {device}")

    # 关键超参数 1：batch_size
    # - 大一点：训练更快，但可能占用更多显存
    # - 小一点：更省资源，但训练更慢
    train_loader, test_loader = build_dataloaders(batch_size=128)

    model = SimpleNN().to(device)

    criterion = nn.CrossEntropyLoss()

    # 关键超参数 2：学习率 lr
    # - 太大：可能震荡不收敛
    # - 太小：收敛慢
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 关键超参数 3：epochs
    # - 太少：欠拟合
    # - 太多：可能过拟合
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer
        )
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    # 训练后可视化预测结果，直观检查模型是否学会识别
    show_predictions(model, test_loader, device, num_images=10)


if __name__ == "__main__":
    main()
