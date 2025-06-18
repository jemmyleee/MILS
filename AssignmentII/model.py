# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- Configuration (can be imported from dataset.py or defined here) ---
NUM_DETECTION_CLASSES = 10 # Should match dataset.py
NUM_SEGMENTATION_CLASSES = 21 # Should match dataset.py
NUM_CLASSIFICATION_CLASSES = 10 # Should match dataset.py
IMG_HEIGHT = 512
IMG_WIDTH = 512

# --- Helper Modules ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- Backbone ---
class EfficientNetB0Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0Backbone, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = efficientnet_b0(weights=weights)
        
        self.features = effnet.features
        self.out_channels = 1280 # For EfficientNet-B0, last feature map channels

    def forward(self, x):
        return self.features(x) # Output feature map, e.g., [B, 1280, 16, 16] for 512x512 input

# --- Neck ---
class SimpleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(SimpleNeck, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x 

# --- Unified Head ---
class UnifiedHead(nn.Module):
    def __init__(self, in_channels, num_det_classes, num_seg_classes, num_cls_classes, num_anchors=3):
        super(UnifiedHead, self).__init__()
        
        self.shared_conv1 = ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1)
        self.shared_conv2 = ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Detection branch output channels: num_anchors * (4 for bbox_xywh + 1 for obj_conf + num_det_classes)
        self.det_output_channels = num_anchors * (4 + 1 + num_det_classes)
        self.det_predictor = nn.Conv2d(in_channels, self.det_output_channels, kernel_size=3, padding=1)

        self.seg_predictor = nn.Conv2d(in_channels, num_seg_classes, kernel_size=1)
        # Upsampling for segmentation will be to the model's input image size
        self.seg_upsample = nn.Upsample(size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_predictor = nn.Linear(in_channels, num_cls_classes)

    def forward(self, x):
        x = self.shared_conv1(x)
        shared_features = self.shared_conv2(x)

        det_out = self.det_predictor(shared_features) 
        
        seg_logits_raw = self.seg_predictor(shared_features)
        seg_out = self.seg_upsample(seg_logits_raw)

        cls_pooled = self.cls_pool(shared_features)
        cls_pooled = torch.flatten(cls_pooled, 1)
        cls_out = self.cls_predictor(cls_pooled)

        return det_out, seg_out, cls_out


# --- Main Multi-Task Model ---
class UnifiedMultiTaskNet(nn.Module):
    def __init__(self, num_det_classes=NUM_DETECTION_CLASSES, 
                 num_seg_classes=NUM_SEGMENTATION_CLASSES, 
                 num_cls_classes=NUM_CLASSIFICATION_CLASSES,
                 num_anchors_det=3, 
                 effnet_pretrained=True):
        super(UnifiedMultiTaskNet, self).__init__()
        
        self.backbone = EfficientNetB0Backbone(pretrained=effnet_pretrained)
        
        self.neck = SimpleNeck(in_channels=self.backbone.out_channels, 
                               mid_channels=256, 
                               out_channels=256) 
        
        self.head = UnifiedHead(in_channels=self.neck.out_channels,
                                num_det_classes=num_det_classes,
                                num_seg_classes=num_seg_classes,
                                num_cls_classes=num_cls_classes,
                                num_anchors=num_anchors_det)
        
        # Store num_anchors for use in loss calculation/postprocessing if needed externally
        self.num_anchors = num_anchors_det
        self.num_det_classes = num_det_classes


    def forward(self, x):
        features = self.backbone(x)        
        neck_out = self.neck(features)      
        det_out, seg_out, cls_out = self.head(neck_out)
        return det_out, seg_out, cls_out

# --- Test model structure and param count ---
if __name__ == '__main__':
    model = UnifiedMultiTaskNet(effnet_pretrained=False) 
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Total Params: {total_params / 1e6:.2f}M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f}M")

    dummy_input = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH) 
    try:
        det_output, seg_output, cls_output = model(dummy_input)
        
        print("\n--- Output Shapes ---")
        # For 512x512 input, EfficientNetB0 stride 32 -> 16x16 feature map
        # det_output: [B, num_anchors * (5 + NUM_DETECTION_CLASSES), H_feat, W_feat]
        # e.g., [2, 3*(4+1+10), 16, 16] = [2, 45, 16, 16]
        print(f"Detection output shape: {det_output.shape}") 
        
        print(f"Segmentation output shape: {seg_output.shape}")
        
        print(f"Classification output shape: {cls_output.shape}")

        if total_params < 8_000_000: #
            print("\nModel parameter count is within the 8M limit.")
        else:
            print("\nWARNING: Model parameter count EXCEEDS 8M limit!")

    except Exception as e:
        print(f"Error during model forward pass test: {e}")