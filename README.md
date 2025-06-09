模型结构详解
1. 多层感知机 (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(512, 256)    # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(256, 128)    # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(128, 10)     # 隐藏层3到输出层
        self.dropout = nn.Dropout(0.3)    # 防止过拟合
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平图像
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

特点：
•	简单全连接网络，忽略图像空间结构
•	参数量：约 40 万，训练速度快
•	准确率：约 97%，受限于模型容量
  
2. 卷积神经网络 (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 第一个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二个卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，降维
        self.fc1 = nn.Linear(64*7*7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层
        self.dropout = nn.Dropout(0.4)  # 防止过拟合
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # 调整维度为[batch, channels, height, width]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # 第一次下采样，14x14
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # 第二次下采样，7x7
        x = x.view(-1, 64*7*7)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

特点：

•	采用卷积层自动提取空间特征
•	参数量：约 15 万，具有参数共享特性
•	准确率：约 99%，适合图像任务的经典架构

 
 
3. 长短期记忆网络 (LSTM)
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(28, 128, batch_first=True, bidirectional=False)  # 第一个LSTM层
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=False)   # 第二个LSTM层
        self.fc1 = nn.Linear(64, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 10)  # 输出层
        self.dropout = nn.Dropout(0.3)  # 防止过拟合
        
    def forward(self, x):
        x = x.view(-1, 28, 28)  # 调整为[batch_size, seq_len, input_size]
        x, _ = self.lstm1(x)    # 输出形状：[batch_size, seq_len, hidden_size]
        x, _ = self.lstm2(x)    # 输出形状：[batch_size, seq_len, hidden_size]
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

特点：
•	将图像按行处理，捕捉行之间的依赖关系
•	参数量：约 5 万，适合序列特征挖掘
•	准确率：约 98%，在图像任务中表现稍逊于 CNN
  
4. 注意力机制模型
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.lstm = nn.LSTM(28, 64, batch_first=True, bidirectional=True)  # 双向LSTM
        self.attention = nn.Sequential(
            nn.Linear(128, 64),  # 将双向LSTM的输出投影到低维空间
            nn.Tanh(),           # 非线性激活
            nn.Linear(64, 1),    # 生成注意力权重
            nn.Softmax(dim=1)    # 对时间步维度进行归一化
        )
        self.fc1 = nn.Linear(128, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 10)   # 输出层
        self.dropout = nn.Dropout(0.3)  # 防止过拟合
        
    def forward(self, x):
        x = x.view(-1, 28, 28)  # 调整为[batch_size, seq_len, input_size]
        x, _ = self.lstm(x)     # 输出形状：[batch_size, seq_len, hidden_size*2]
        attn_weights = self.attention(x)  # 生成注意力权重
        x = torch.sum(x * attn_weights, dim=1)  # 加权求和
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

特点：
•	结合双向 LSTM 和注意力机制，动态关注重要行
•	参数量：约 4 万，轻量化且性能优良
•	准确率：约 99%，通过注意力机制有效提升识别能力
  
调参过程概述
针对 MNIST 手写数字识别任务，我对四个模型（MLP、CNN、LSTM、Attention）进行了系统调参。调参过程遵循以下策略：
1.	基础参数设置：
o	初始学习率：0.001（Adam 优化器）
o	批次大小：64
o	训练轮数：5-20（根据收敛情况调整）
o	优化器：Adam（自适应学习率）
2.	调参顺序：
o	首先调整学习率和批次大小
o	然后调整模型特定参数（如 CNN 的卷积核数量）
o	最后优化正则化参数（dropout 率）
3.	评估指标：
o	验证集准确率
o	训练时间
o	过拟合程度（训练集与验证集准确率差距）

模型特定处理
•	MLP：展平为一维向量 (28×28=784)
•	CNN：保持二维图像结构 (1×28×28)
•	RNN/LSTM：将图像视为序列 (28 个时间步，每步 28 个特征)
•	注意力机制：与 RNN 相同输入，但通过注意力权重动态聚焦关键特征
