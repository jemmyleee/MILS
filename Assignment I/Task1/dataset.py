import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (str): 包含图像路径和标签的txt文件路径。
            root_dir (str): 数据集根目录。
            transform (callable, optional): 图像变换。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 读取txt文件，解析路径和标签
        with open(txt_file, 'r') as f:
            self.data = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_path)  # 拼接完整路径
        label = int(label)

        # 加载图像并应用变换
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class MultiChannelMiniImageNetDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, channels='rgb'):
        """
        支持不同通道組合的MiniImageNet數據集
        
        參數:
            txt_file (str): 包含圖像路徑和標籤的txt文件路徑
            root_dir (str): 數據集根目錄
            transform (callable, optional): 圖像變換
            channels (str): 要使用的通道，可以是'rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels.lower()
        
        # 驗證通道設置
        valid_channels = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
        if self.channels not in valid_channels:
            raise ValueError(f"通道必須是以下之一: {valid_channels}")
        
        # 讀取txt文件，解析路徑和標籤
        with open(txt_file, 'r') as f:
            self.data = [line.strip().split() for line in f.readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_path)
        label = int(label)
        
        # 加載圖像
        image = Image.open(img_path).convert('RGB')
        
        # 應用變換
        if self.transform:
            image = self.transform(image)
        
        # 根據指定的通道選擇
        if self.channels == 'rgb':
            return image, label
        elif self.channels == 'r':
            return image[0:1], label
        elif self.channels == 'g':
            return image[1:2], label
        elif self.channels == 'b':
            return image[2:3], label
        elif self.channels == 'rg':
            return image[0:2], label
        elif self.channels == 'rb':
            return torch.cat([image[0:1], image[2:3]]), label
        elif self.channels == 'gb':
            return image[1:3], label


