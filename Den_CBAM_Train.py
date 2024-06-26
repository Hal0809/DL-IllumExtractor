import torch
import torch.nn as nn
import os
from torchvision import transforms
from DatasetLoader import CustomDataset
from torch.utils.data import DataLoader
from arch.densenet201_CBAM import DenseNet_CBAM
import datetime
import pandas as pd

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练集
train_data_dir = os.path.join('data', 'SimpleCube++', 'train', 'PNG')
train_csv_file = os.path.join('data', 'SimpleCube++', 'train', 'gt.csv')
# 创建训练集实例
train_dataset = CustomDataset(train_data_dir, train_csv_file, transform=transform)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义测试集
test_data_dir = os.path.join('data', 'SimpleCube++', 'test', 'PNG')
test_csv_file = os.path.join('data', 'SimpleCube++', 'test', 'gt.csv')
# 创建测试集实例
test_dataset = CustomDataset(test_data_dir, test_csv_file, transform=transform)
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model = DenseNet_CBAM(pretrained=False)
model.load_state_dict(torch.load(os.path.join('models', 'Den_.pth'), map_location=device), strict=False)
model.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
results = []
num_epochs = 24
time_start = datetime.datetime.now()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        labels = labels.type(torch.float32)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    # 在验证集上评估模型性能
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            validation_loss += loss.item() * inputs.size(0)
        validation_loss /= len(test_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss}, Test Loss: {validation_loss}, Time: {datetime.datetime.now()-time_start}")
    results.append([epoch+1, epoch_loss, validation_loss, datetime.datetime.now()-time_start])

# 保存模型的参数
torch.save(model.state_dict(), os.path.join('models', 'Den_CBAM.pth'))
csvout = pd.DataFrame(results, columns=['epoch', 'epoch_loss', 'validation_loss', 'time'])
csvout.to_csv(os.path.join('results', 'den_CBAM.csv'))
print("Training task finished in {}".format(datetime.datetime.now()-time_start))
