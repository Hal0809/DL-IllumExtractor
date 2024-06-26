import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data_info = pd.read_csv(csv_file)  # 从CSV文件读取数据信息

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data_info.iloc[idx, 0] + '.png')  # 文件名列
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([self.data_info.iloc[idx, 1], self.data_info.iloc[idx, 2]])  # 标签列
        return image, label
