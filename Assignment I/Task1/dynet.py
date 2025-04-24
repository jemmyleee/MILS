import torch
import torch.nn as nn

class DynamicConv2d(nn.Module):
    def __init__(self, max_in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, K=4, temperature=30):
        """
        動態卷積模組，可處理不同數量的輸入通道
        
        參數:
            max_in_channels (int): 最大可能的輸入通道數
            out_channels (int): 輸出通道數
            kernel_size (int or tuple): 卷積核大小
            stride (int or tuple): 步長
            padding (int or tuple): 填充
            bias (bool): 是否使用偏置
            K (int): 平行卷積核的數量
            temperature (float): softmax溫度參數
        """
        super().__init__()
        self.max_in_channels = max_in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.K = K
        self.temperature = temperature
        
        # 創建K個平行卷積核
        self.kernels = nn.ModuleList([
            nn.Conv2d(max_in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            for _ in range(K)
        ])
        
        # 注意力機制 - 用於動態選擇卷積核的權重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(max_in_channels, max(max_in_channels // 4, 1), 1, bias=False),  # 降維
            nn.ReLU(inplace=True),
            nn.Conv2d(max(max_in_channels // 4, 1), K, 1, bias=False)  # 產生K個注意力權重
        )
        
    def forward(self, x):
        # 處理輸入通道數不足的情況
        b, c, h, w = x.size()
        if c < self.max_in_channels:
            # 填充缺失的通道為0
            padding = torch.zeros(b, self.max_in_channels - c, h, w, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # 計算注意力權重
        attn = self.attention(x)
        attn = attn.view(b, self.K, 1, 1, 1) # 為了之後能乘
        
        # 使用softmax生成歸一化的注意力權重，加入溫度參數
        attn = torch.softmax(attn / self.temperature, dim=1)
        
        # 應用每個卷積核並根據注意力權重聚合結果
        out = 0
        for i, conv in enumerate(self.kernels):
            out = out + conv(x) * attn[:, i]
        
        return out

class DynamicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, K=4, temperature=30):
        super().__init__()
        self.conv1 = DynamicConv2d(in_channels, out_channels, 3, stride, padding=1, bias=False, K=K, temperature=temperature)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DynamicConv2d(out_channels, out_channels, 3, padding=1, bias=False, K=K, temperature=temperature)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                DynamicConv2d(in_channels, out_channels, 1, stride, bias=False, K=K, temperature=temperature),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class DynamicResNet12(nn.Module):
    def __init__(self, num_classes=100, K=4, temperature=30):
        super().__init__()
        self.in_channels = 64
        self.K = K
        self.temperature = temperature
        
        # 第一層使用動態卷積，可處理不同通道輸入
        self.conv = nn.Sequential(
            DynamicConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, K=K, temperature=temperature),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer1 = self._make_layer(64, 3, stride=2)
        self.layer2 = self._make_layer(128, 3, stride=2)
        self.layer3 = self._make_layer(256, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [DynamicResBlock(self.in_channels, out_channels, stride, self.K, self.temperature)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(DynamicResBlock(out_channels, out_channels, K=self.K, temperature=self.temperature))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
