import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 超参数
EPOCH = 100  # 训练的轮数
BATCH_SIZE = 64  # 批大小
LR = 0.005  # 学习率

# 加载CSV数据
df = pd.read_csv('/home/zhanghongyu/panruwei/apkutils-master/apk_info.csv')

# 数据预处理
# 将分类列转换为数值
categorical_columns = ['App Name', 'Package Name', 'Main Activity', 'Activities', 'Services', 'Receivers', 'Permissions', 'MD5', 'Logo Path', 'Cert_SHA1', 'Cert_SHA256', 'Cert_Issuer', 'Cert_Subject', 'Cert_Hash_Algo', 'Cert_Signature_Algo', 'Cert_Serial_Number']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 标准化/归一化数据
scaler = StandardScaler()
df[categorical_columns] = scaler.fit_transform(df[categorical_columns])

# 分割数据集
X = df[categorical_columns].values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=X_train.shape[1],  # 输入特征的维度
            hidden_size=64,  # LSTM隐藏单元的大小
            num_layers=1,  # LSTM层数
            batch_first=True,  # 输入和输出的第一个维度是batch size
        )
        self.out = nn.Linear(64, len(set(y)))  # 输出层与分类类别数量匹配

    def forward(self, x):
        x = x.unsqueeze(1)  # 在batch_first模式下需要添加额外的维度
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练和测试循环
for epoch in range(EPOCH):
    epoch_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{EPOCH}') as pbar:
        for step, (b_x, b_y) in enumerate(train_loader):
            output = rnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()

            pbar.set_postfix({'loss': epoch_loss / (step + 1), 'accuracy': 100. * correct / total})
            pbar.update(1)

    with torch.no_grad():
        test_correct = 0
        for test_x, test_y in test_loader:
            test_output = rnn(test_x)
            _, pred_y = torch.max(test_output, 1)
            test_correct += (pred_y == test_y).sum().item()
        test_accuracy = 100. * test_correct / len(test_dataset)
        print(f'Epoch: {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f} | Test Accuracy: {test_accuracy:.2f}%')

