import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dynet import DynamicResNet12
from dataset import MultiChannelMiniImageNetDataset
from thop import profile, clever_format
import os
import sys
import contextlib

# ====== 工具：靜音 thop 的 print 輸出 ======
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ====== 損失函數 (用不到可以直接忽略) ======
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ====== 主測試函數 ======
def run_test(channels='rgb', K=4, temperature=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 讀取模型
    model = DynamicResNet12(num_classes=100, K=K, temperature=temperature).to(device)
    model_path = f"experiment/best_model_{channels}_K{K}_temp{temperature}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 數據處理
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MultiChannelMiniImageNetDataset(
        txt_file='../test.txt', 
        root_dir='../',
        transform=transform,
        channels=channels
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 測 FLOPs 和參數量
    if channels == 'rgb':
        input_channels = 3
    elif channels in ['r', 'g', 'b']:
        input_channels = 1
    else:
        input_channels = 2

    dummy_input = torch.randn(1, input_channels, 84, 84).to(device)
    with suppress_stdout():
        macs, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([macs * 2, params], '%.3f')

    # 測試
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Testing {channels}'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    test_acc = 100. * correct / total

    return test_acc, flops, params

# ====== 批次測試所有通道組合 ======
if __name__ == "__main__":
    K = 4
    temperature = 30

    results = {}

    for channels in ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']:
        print(f"\n{'='*50}")
        print(f"開始測試 {channels.upper()} 通道")
        print(f"{'='*50}")

        acc, flops, params = run_test(channels=channels, K=K, temperature=temperature)
        results[channels] = (acc, flops, params)

    # 顯示結果總結
    print("\n\n最終結果總結:")
    print("="*60)
    print(f"{'通道':<10} {'準確率(%)':<12} {'FLOPs':<15} {'參數量':<15}")
    print("="*60)
    for channels, (acc, flops, params) in results.items():
        print(f"{channels.upper():<10} {acc:.2f}{'%':<8} {flops:<15} {params:<15}")
