import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from net import ResNet34,WideCNN4
from dataset import MiniImageNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # 新增TensorBoard支持
from tqdm import tqdm  # 新增tqdm支持
import torchvision  # Import torchvision for utilities like make_grid
import datetime
import torch.nn.functional as F


# 图像变换（标准化和数据增强）
transform_train = transforms.Compose([
    transforms.Resize(84),
    transforms.RandomCrop(84, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_dataset = MiniImageNetDataset(txt_file='../test.txt', root_dir='../', transform=transform_val_test)

# 创建DataLoader

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

import torch
from thop import profile, clever_format
from tqdm import tqdm

# 選擇設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = WideCNN4(num_classes=100, width=256).to(device)
model2 = ResNet34(num_classes=100).to(device)

# 載入已訓練好的權重
model.load_state_dict(torch.load("experiment/best_model_my.pth", map_location=device, weights_only=True))
model2.load_state_dict(torch.load("experiment/best_model.pth", map_location=device, weights_only=True))

# 印出模型 FLOPs 和 參數量
dummy_input = torch.randn(1, 3, 84, 84).to(device)
for name, net in [("MyNet", model), ("ResNet34", model2)]:
    macs, params = profile(net, inputs=(dummy_input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"📊 {name} - FLOPs: {macs}, Params: {params}")

# 測試模式
model.eval()
model2.eval()

correct = 0
correct2 = 0

# 開始測試
with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Testing'):
        data, target = data.to(device), target.to(device)

        output = model(data)
        output2 = model2(data)

        pred = output.argmax(dim=1)
        pred2 = output2.argmax(dim=1)

        correct += pred.eq(target).sum().item()
        correct2 += pred2.eq(target).sum().item()

# 計算準確率
test_acc = 100. * correct / len(test_loader.dataset)
test_acc2 = 100. * correct2 / len(test_loader.dataset)

print(f"\n🎯 MyNet Test Accuracy: {test_acc:.2f}%")
print(f"🎯 ResNet34 Test Accuracy: {test_acc2:.2f}%")
