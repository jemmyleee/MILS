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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
# 初始化TensorBoard Writer
log_dir = f"runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet34(num_classes=100).to(device)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

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

# 加载数据集
train_dataset = MiniImageNetDataset(txt_file='../train.txt', root_dir='../', transform=transform_train)
val_dataset = MiniImageNetDataset(txt_file='../val.txt', root_dir='../', transform=transform_val_test)
test_dataset = MiniImageNetDataset(txt_file='../test.txt', root_dir='../', transform=transform_val_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

def log_weights(writer, model, epoch):
    """记录权重直方图到TensorBoard"""
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, epoch)

def train_epoch(epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)  # 初始化tqdm进度条
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # 记录第一批样本图像
        if epoch == 0 and batch_idx == 0:
            img_grid = torchvision.utils.make_grid(data[:8])
            writer.add_image('train_images', img_grid, 0)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 实时更新进度条信息
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.4f}"
        })
        
        # 记录每batch的损失
        writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    log_weights(writer, model, epoch)  # 记录权重分布
    return avg_loss

def validate():
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)
    return accuracy, avg_loss

# 主训练循环
best_acc = 0
for epoch in range(1, 101):
    train_loss = train_epoch(epoch)
    val_acc, val_loss = validate()
    
    # 记录验证指标
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 打印带颜色的进度信息
    print(f"\nEpoch {epoch:03d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "experiment/best_model.pth")
        print(f"🔥 New best model saved (Acc: {val_acc:.2f}%)")

# 最终测试
model.load_state_dict(torch.load("experiment/best_model.pth", weights_only=True))
model.eval()
correct = 0
with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Testing'):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_acc = 100. * correct / len(test_loader.dataset)
writer.add_scalar('Accuracy/test', test_acc, 0)
writer.close()  # 关闭TensorBoard写入器

print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
