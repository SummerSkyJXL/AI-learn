# MNIST 神经网络学习指南（对应 `.vscode/learn.py`）

## 1. 代码地图与阅读顺序
推荐阅读顺序：
1. `main()`：先看主流程如何把各模块串起来
2. `build_dataloaders()`：理解数据从哪里来、如何预处理
3. `SimpleNN`：看模型结构（输入维度、隐藏层、输出维度）
4. `train_one_epoch()`：理解训练闭环 5 步
5. `evaluate()`：理解为什么评估时不用梯度
6. `show_predictions()`：看预测可视化和反标准化

## 2. 核心概念词典
- `batch`：一次喂给模型的一小批样本。
- `epoch`：完整看一遍训练集。
- `logits`：模型输出的原始分数（还没 softmax）。
- `loss`：损失函数值，衡量预测与真实标签差距。
- `backward`：反向传播，计算参数梯度。
- `optimizer.step()`：根据梯度更新参数。

## 3. 一次完整训练流程（单 batch）
在 `train_one_epoch()` 中：
1. `optimizer.zero_grad()`：清空上一批次残留梯度
2. `logits = model(images)`：前向传播
3. `loss = criterion(logits, labels)`：计算损失
4. `loss.backward()`：反向传播算梯度
5. `optimizer.step()`：参数更新

你可以在调试器里重点看这几个变量：
- `images.shape`（期望 `[B,1,28,28]`）
- `logits.shape`（期望 `[B,10]`）
- `loss.item()`（是否逐步下降）

## 4. 常见报错与排查
- `ModuleNotFoundError: No module named 'torchvision'`
  - 解释器不是 `myenv312`，或包没装在当前环境。
- `import` 波浪线无法解析
  - Cursor 解释器未切到 `/Users/jiaoxulin/miniconda3/envs/myenv312/bin/python`。
- `Matplotlib cache directory not writable`
  - 不影响训练结果；需要时可设置 `MPLCONFIGDIR` 到可写目录。
- 训练很慢
  - 检查 `print(device)` 是否是 `mps`；如果是 `cpu`，通常会慢很多。

## 5. 下一步进阶路线
1. 把 MLP 改为 CNN（`Conv2d + MaxPool2d`），观察精度变化。
2. 尝试数据增强（如随机旋转）理解泛化能力。
3. 加入模型保存与加载：`torch.save` / `load_state_dict`。
4. 尝试学习率、batch size、epoch 对收敛速度的影响。

## 6. Cursor 插件工具链（完整学习版）

### A. 解释与补全（核心）
- Python（Microsoft）
- Pylance（Microsoft）
- Jupyter（Microsoft）

开启后第一件事：
1. 选择解释器为 `myenv312`。
2. 在 `learn.py` 悬停变量（如 `images`、`logits`）看类型提示。
3. 使用 `Go to Definition` 跳转到 `nn.Linear` 等 API。

### B. 调试与可视化
- Python Debugger（如果你的 Python 扩展未内置调试器则单独安装）
- Data Wrangler（Microsoft）

开启后第一件事：
1. 在 `train_one_epoch()` 的 `logits = model(images)` 行打断点。
2. 运行调试，观察 `images.shape`、`logits.shape`、`labels[:8]`。
3. 用 Data Wrangler 打开导出的样例数据（可选）看分布。

### C. AI 学习辅助（用 Cursor 内置 Chat）
推荐提示模板（选中代码后提问）：
- 「请按张量形状逐行解释这段代码，每行输出输入 shape 与输出 shape。」
- 「这段训练代码里如果去掉 `zero_grad()` 会发生什么？给出直观例子。」
- 「把这段代码改写成更容易教学的版本，但保持逻辑不变。」

## 7. 最小自测清单
- 代码能跑完 5 个 epoch。
- `Test Acc` 达到约 97%（波动属正常）。
- 能解释清楚 `logits` 和 `loss` 的关系。
- 能口头复述训练闭环 5 步。
