import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from Task1.dynet import DynamicResNet12
from Task1.dataset import MultiChannelMiniImageNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import torch.nn.functional as F
import torchvision

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

def run_experiment(channels='rgb', K=4, temperature=30):
    # 設置實驗名稱
    exp_name = f"dynamic_conv_K{K}_temp{temperature}_{channels}"
    log_dir = f"runs/{exp_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 創建動態卷積模型
    model = DynamicResNet12(num_classes=100, K=K, temperature=temperature).to(device)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # 圖像變換
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
    
    # 加載數據集，使用指定通道
    train_dataset = MultiChannelMiniImageNetDataset(
        txt_file='./train.txt', 
        root_dir='./', 
        transform=transform_train,
        channels=channels
    )
    
    val_dataset = MultiChannelMiniImageNetDataset(
        txt_file='./val.txt', 
        root_dir='./', 
        transform=transform_val_test,
        channels=channels
    )
    
    test_dataset = MultiChannelMiniImageNetDataset(
        txt_file='./test.txt', 
        root_dir='./', 
        transform=transform_val_test,
        channels=channels
    )
    
    # 創建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def train_epoch(epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # 記錄第一批樣本圖像
            if epoch == 0 and batch_idx == 0:
                if data.size(1) == 1:
                    # 對於單通道圖像，複製到3通道以便可視化
                    img_grid = torchvision.utils.make_grid(data[:8].repeat(1, 3, 1, 1))
                elif data.size(1) == 2:
                    # 對於雙通道圖像，添加一個零通道
                    zeros = torch.zeros_like(data[:8, 0:1])
                    three_channel = torch.cat([data[:8], zeros], dim=1)
                    img_grid = torchvision.utils.make_grid(three_channel)
                else:
                    img_grid = torchvision.utils.make_grid(data[:8])
                writer.add_image(f'train_images_{channels}', img_grid, 0)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 計算準確率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 實時更新進度條信息
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.4f}"
            })
            
            # 記錄每batch的損失
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        return avg_loss, train_acc
    
    def validate():
        model.eval()
        correct = 0
        total_loss = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return accuracy, avg_loss
    
    # 主訓練循環
    best_acc = 0
    for epoch in range(1, 101):
        train_loss, train_acc = train_epoch(epoch)
        val_acc, val_loss = validate()
        
        # 記錄驗證指標
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 打印進度信息
        print(f"\nEpoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{channels}_K{K}_temp{temperature}.pth")
            print(f"🔥 New best model saved (Acc: {val_acc:.2f}%)")
    
    # 最終測試
    model.load_state_dict(torch.load(f"best_model_{channels}_K{K}_temp{temperature}.pth"))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * correct / total
    writer.add_scalar('Accuracy/test', test_acc, 0)
    
    print(f"\n🎯 Final Test Accuracy ({channels}): {test_acc:.2f}%")
    writer.close()
    
    return test_acc

# 執行所有通道組合的實驗
if __name__ == "__main__":
    # 設置動態卷積參數
    K = 4  # 平行卷積核數量
    temperature = 30  # softmax溫度參數
    
    # 記錄所有通道組合的結果
    results = {}
    
    # 對所有通道組合進行實驗
    for channels in ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']:
        print(f"\n\n{'='*50}")
        print(f"開始 {channels} 通道實驗")
        print(f"{'='*50}\n")
        
        acc = run_experiment(channels=channels, K=K, temperature=temperature)
        results[channels] = acc
    
    # 打印所有結果
    print("\n\n最終結果匯總:")
    print("="*30)
    for channels, acc in results.items():
        print(f"{channels.upper()} 通道: {acc:.2f}%")
