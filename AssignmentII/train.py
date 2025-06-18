# train.py
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset # Subset 用於創建 Replay Buffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy # For LwF teacher model
import torch.nn.functional as F

from dataset import (
    get_transform, get_segmentation_transforms,
    COCODetectionDataset, VOCSegmentationDataset, ImagenetteClassificationDataset,
    IMG_HEIGHT, IMG_WIDTH,
    NUM_DETECTION_CLASSES, NUM_SEGMENTATION_CLASSES, NUM_CLASSIFICATION_CLASSES
)
from model import UnifiedMultiTaskNet

import torchmetrics
import torchvision # For NMS
from torchvision.ops.boxes import distance_box_iou

# --- Anchor Configuration (Global for simplicity) ---
ANCHOR_BOXES = [ 
    (0.5, 1.0),
    (1.0, 0.5),
    (1.0, 1.0),
]
# --- Helper Functions & Configuration ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, lr, optimizer_name="adamw"):
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

def get_scheduler(optimizer, scheduler_name="cosine", total_epochs=100, milestones=None):
    if scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif scheduler_name.lower() == "step":
        milestones = milestones if milestones else [int(0.6*total_epochs), int(0.8*total_epochs)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        return None

# --- Loss Functions ---
seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
cls_criterion = nn.CrossEntropyLoss()

def intersection_over_union(boxes_preds, boxes_labels, box_format="xyxy"):
    if box_format == "cxcywh":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "xyxy":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, anchors_config, feature_map_stride=32, iou_threshold_pos=0.5, iou_threshold_neg=0.4):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss() 
        self.ce = nn.CrossEntropyLoss() 
        self.num_classes = num_classes
        self.anchors_config = torch.tensor(anchors_config, dtype=torch.float32)
        self.num_anchors = len(anchors_config)
        self.stride = feature_map_stride
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg

    def forward(self, predictions, targets, current_img_size=(IMG_HEIGHT, IMG_WIDTH)):
        device = predictions.device
        batch_size, _, H_feat, W_feat = predictions.shape
        
        predictions_reshaped = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, H_feat, W_feat).permute(0, 1, 3, 4, 2).contiguous()
        
        pred_xy_sig = torch.sigmoid(predictions_reshaped[..., 0:2])
        pred_wh_exp = torch.exp(predictions_reshaped[..., 2:4]) * self.anchors_config.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        pred_conf_logits = predictions_reshaped[..., 4:5] 
        pred_cls_logits = predictions_reshaped[..., 5:]   

        grid_x = torch.arange(W_feat, device=device, dtype=torch.float32).repeat(H_feat, 1).unsqueeze(0)
        grid_y = torch.arange(H_feat, device=device, dtype=torch.float32).repeat(W_feat, 1).t().unsqueeze(0)
        
        abs_pred_x = (pred_xy_sig[..., 0] + grid_x) * self.stride
        abs_pred_y = (pred_xy_sig[..., 1] + grid_y) * self.stride
        abs_pred_w = pred_wh_exp[..., 0] * self.stride
        abs_pred_h = pred_wh_exp[..., 1] * self.stride
        
        pred_boxes_xyxy_batch = torch.stack([
            abs_pred_x - abs_pred_w / 2,
            abs_pred_y - abs_pred_h / 2,
            abs_pred_x + abs_pred_w / 2,
            abs_pred_y + abs_pred_h / 2
        ], dim=-1)

        loc_loss_batch_accum = torch.tensor(0.0, device=device)
        conf_loss_batch_accum = torch.tensor(0.0, device=device)
        cls_loss_batch_accum = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            target_boxes_item = targets[b]["boxes"].to(device) 
            target_labels_item = targets[b]["labels"].to(device) 

            pred_boxes_item_flat = pred_boxes_xyxy_batch[b].view(-1, 4) 
            pred_conf_item_flat_logits = pred_conf_logits[b].view(-1, 1)
            pred_cls_item_logits_flat_reshaped = pred_cls_logits[b].view(-1, self.num_classes)

            # --- 新增檢查：如果此圖像沒有真實框 ---
            if target_boxes_item.shape[0] == 0: 
                # 只有 conf loss，所有預測都應該是背景
                conf_loss_no_gt = self.bce_logits(pred_conf_item_flat_logits, torch.zeros_like(pred_conf_item_flat_logits))
                conf_loss_batch_accum += conf_loss_no_gt
                # print(f"Batch item {b}: No ground truth boxes. Skipping loc and cls loss.")
                continue # 直接處理下一個 batch item
            # --- 檢查結束 ---
            
            # ious shape: (N_all_anchors_in_item, num_target_boxes_in_item)
            ious = intersection_over_union(pred_boxes_item_flat.unsqueeze(1), target_boxes_item.unsqueeze(0), box_format="xyxy")
            best_gt_iou_for_anchor, best_gt_idx_for_anchor = ious.max(dim=1) 
            
            pos_mask_flat = (best_gt_iou_for_anchor >= self.iou_threshold_pos)
            neg_mask_flat = (best_gt_iou_for_anchor < self.iou_threshold_neg)

            if pos_mask_flat.ndim == 2 and pos_mask_flat.shape[-1] == 1:
                pos_mask_flat = pos_mask_flat.squeeze(-1)
            if neg_mask_flat.ndim == 2 and neg_mask_flat.shape[-1] == 1:
                neg_mask_flat = neg_mask_flat.squeeze(-1)

            target_obj_conf = torch.zeros_like(pred_conf_item_flat_logits) 
            target_obj_conf[pos_mask_flat] = 1.0 
            
            conf_mask_for_loss = (pos_mask_flat | neg_mask_flat) 
            
            if conf_mask_for_loss.sum() > 0:
                 conf_loss_batch_accum += self.bce_logits(
                     pred_conf_item_flat_logits[conf_mask_for_loss], 
                     target_obj_conf[conf_mask_for_loss]            
                 )
            
            num_pos_item = pos_mask_flat.sum().item()

            if num_pos_item == 0: 
                continue

            pred_boxes_pos = pred_boxes_item_flat[pos_mask_flat] 
            
            # 获取匹配的 GT boxes 和 labels
            matched_gt_indices = best_gt_idx_for_anchor[pos_mask_flat]
            matched_gt_boxes = target_boxes_item[matched_gt_indices] 
            matched_gt_labels = target_labels_item[matched_gt_indices]

            # --- 新增：確保 matched_gt_boxes 和 matched_gt_labels 的形狀 ---
            # matched_gt_boxes 應該是 [N_pos_item, 4]
            if matched_gt_boxes.ndim == 3 and matched_gt_boxes.shape[1] == 1:
                # print(f"Squeezing matched_gt_boxes from {matched_gt_boxes.shape}")
                matched_gt_boxes = matched_gt_boxes.squeeze(1)
            
            # matched_gt_labels 應該是 [N_pos_item]
            if matched_gt_labels.ndim == 2 and matched_gt_labels.shape[1] == 1:
                # print(f"Squeezing matched_gt_labels from {matched_gt_labels.shape}")
                matched_gt_labels = matched_gt_labels.squeeze(1)
            elif matched_gt_labels.ndim > 1: # 如果超過一維，則嘗試展平
                # print(f"Warning: matched_gt_labels has unexpected ndim {matched_gt_labels.ndim}. Flattening. Shape was {matched_gt_labels.shape}")
                matched_gt_labels = matched_gt_labels.flatten()

            # --- 確保結束 ---
            
            # print(f"Shape after potential squeeze: matched_gt_boxes: {matched_gt_boxes.shape}, matched_gt_labels: {matched_gt_labels.shape}")


            if pred_boxes_pos.numel() > 0 and matched_gt_boxes.numel() > 0:
                if pred_boxes_pos.shape[1] != 4 or matched_gt_boxes.shape[1] != 4:
                    print(f"CRITICAL SHAPE ERROR before diou_matrix for item {b}:")
                    print(f"pred_boxes_pos shape: {pred_boxes_pos.shape}")
                    print(f"matched_gt_boxes shape: {matched_gt_boxes.shape}")
                    if pred_boxes_pos.shape[0] > 0 and pred_boxes_pos.shape[1] != 4 : pred_boxes_pos = torch.empty((0,4), device=device) 
                    if matched_gt_boxes.shape[0] > 0 and matched_gt_boxes.shape[1] !=4 : matched_gt_boxes = torch.empty((0,4), device=device)


                if pred_boxes_pos.shape[0] > 0 and matched_gt_boxes.shape[0] > 0 and \
                   pred_boxes_pos.shape[1] == 4 and matched_gt_boxes.shape[1] == 4:
                    # 確保 pred_boxes_pos 和 matched_gt_boxes 的 N_pos_item 相同
                    if pred_boxes_pos.shape[0] != matched_gt_boxes.shape[0]:
                        print(f"CRITICAL MISMATCH N_pos for item {b}: pred_boxes_pos: {pred_boxes_pos.shape[0]}, matched_gt_boxes: {matched_gt_boxes.shape[0]}")
                    else:
                        diou_matrix = distance_box_iou(pred_boxes_pos, matched_gt_boxes)
                        diou_values = torch.diag(diou_matrix)
                        loc_loss_item = (1.0 - diou_values).sum() 
                        loc_loss_batch_accum += loc_loss_item
            
            pred_cls_pos_logits = pred_cls_item_logits_flat_reshaped[pos_mask_flat] 
            
            # 再次確保 matched_gt_labels 是一維的
            if matched_gt_labels.ndim != 1:
                 print(f"ERROR: matched_gt_labels is not 1D before CrossEntropy! Shape: {matched_gt_labels.shape}")
                 # 應急處理或拋出錯誤
                 if matched_gt_labels.numel() == pred_cls_pos_logits.shape[0]: # 如果元素數量匹配，嘗試 flatten
                     matched_gt_labels = matched_gt_labels.flatten()
                 else: # 如果元素數量也不匹配，這是一個更嚴重的問題
                     # 在這裡可以選擇跳過這個 batch item 的分類損失，或者拋出錯誤
                     print(f"CRITICAL: Matched_gt_labels ({matched_gt_labels.shape}) numel mismatch with pred_cls_pos_logits ({pred_cls_pos_logits.shape}) for item {b}. Skipping cls_loss for this item.")
                     # cls_loss_batch_accum += torch.tensor(0.0, device=device) # 或者不加
                     pass # 跳過下面的 cls_loss 計算
            
            # 只有在 matched_gt_labels 確實是一維且與 pred_cls_pos_logits 的 batch 維度匹配時才計算
            if matched_gt_labels.ndim == 1 and matched_gt_labels.shape[0] == pred_cls_pos_logits.shape[0]:
                cls_loss_batch_accum += self.ce(pred_cls_pos_logits, matched_gt_labels.long() - 1)
            # else:
                # print(f"Skipping cls_loss for item {b} due to label shape mismatch even after trying to fix.")

        total_loss = loc_loss_batch_accum + conf_loss_batch_accum + cls_loss_batch_accum
        return total_loss, loc_loss_batch_accum, conf_loss_batch_accum, cls_loss_batch_accum


detection_criterion = DetectionLoss(num_classes=NUM_DETECTION_CLASSES, anchors_config=ANCHOR_BOXES)


# --- Evaluation Functions ---
# (evaluate_segmentation, decode_detection_outputs, evaluate_detection, evaluate_classification 保持不變)
@torch.no_grad()
def evaluate_segmentation(model, val_loader, device, writer, epoch, stage_name):
    model.eval()
    mIoU_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=NUM_SEGMENTATION_CLASSES, ignore_index=255).to(device)
    total_loss = 0
    for images, masks in tqdm(val_loader, desc=f"Eval Seg Epoch {epoch+1}", leave=False):
        images, masks = images.to(device), masks.to(device)
        _, seg_outputs, _ = model(images)
        loss = seg_criterion(seg_outputs, masks)
        total_loss += loss.item()
        mIoU_metric.update(seg_outputs.argmax(dim=1), masks)
        
    avg_loss = total_loss / len(val_loader)
    iou = mIoU_metric.compute().item()
    mIoU_metric.reset()
    
    if writer:
        writer.add_scalar(f"{stage_name}/Seg_Val_Loss", avg_loss, epoch)
        writer.add_scalar(f"{stage_name}/Seg_Val_mIoU", iou, epoch)
    print(f"Epoch {epoch+1} Seg Val: Avg Loss: {avg_loss:.4f}, mIoU: {iou:.4f}")
    return iou

def decode_detection_outputs(det_outputs_raw, current_model, conf_threshold=0.25, nms_iou_threshold=0.45):
    device = det_outputs_raw.device
    batch_size, _, H_feat, W_feat = det_outputs_raw.shape
    
    num_anchors = current_model.num_anchors
    num_classes = current_model.num_det_classes
    
    active_anchor_configs_list = ANCHOR_BOXES[:num_anchors]
    anchor_configs = torch.tensor(active_anchor_configs_list, dtype=torch.float32).to(device)
    
    stride = IMG_HEIGHT // H_feat

    predictions = det_outputs_raw.view(batch_size, num_anchors, 5 + num_classes, H_feat, W_feat).permute(0, 1, 3, 4, 2).contiguous()
    
    pred_xy_rel = torch.sigmoid(predictions[..., 0:2])
    pred_wh_rel = torch.exp(predictions[..., 2:4]) * anchor_configs.view(1, num_anchors, 1, 1, 2)

    pred_conf_logits = predictions[..., 4:5]
    pred_cls_logits = predictions[..., 5:]

    pred_conf_scores = torch.sigmoid(pred_conf_logits).squeeze(-1)
    pred_cls_probs = torch.softmax(pred_cls_logits, dim=-1)
    pred_cls_scores, pred_cls_indices = torch.max(pred_cls_probs, dim=-1)

    grid_x = torch.arange(W_feat, device=device).repeat(H_feat, 1).float()
    grid_y = torch.arange(H_feat, device=device).repeat(W_feat, 1).t().float()
    
    abs_pred_x = (pred_xy_rel[..., 0] + grid_x.view(1, 1, H_feat, W_feat)) * stride
    abs_pred_y = (pred_xy_rel[..., 1] + grid_y.view(1, 1, H_feat, W_feat)) * stride
    abs_pred_w = pred_wh_rel[..., 0] * stride
    abs_pred_h = pred_wh_rel[..., 1] * stride
    
    pred_boxes_xyxy = torch.stack([
        abs_pred_x - abs_pred_w / 2,
        abs_pred_y - abs_pred_h / 2,
        abs_pred_x + abs_pred_w / 2,
        abs_pred_y + abs_pred_h / 2
    ], dim=-1)

    output_batch = []
    for b in range(batch_size):
        boxes_img = pred_boxes_xyxy[b].view(-1, 4)
        scores_img = (pred_conf_scores[b] * pred_cls_scores[b]).view(-1)
        labels_img = (pred_cls_indices[b]).view(-1)

        keep_mask = scores_img >= conf_threshold
        boxes_img = boxes_img[keep_mask]
        scores_img = scores_img[keep_mask]
        labels_img = labels_img[keep_mask]

        if boxes_img.numel() == 0:
            output_batch.append({
                'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty((0,), device=device),
                'labels': torch.empty((0,), dtype=torch.long, device=device),
            })
            continue
            
        boxes_img[:, 0::2].clamp_(min=0, max=IMG_WIDTH)
        boxes_img[:, 1::2].clamp_(min=0, max=IMG_HEIGHT)
        
        keep_indices = torchvision.ops.nms(boxes_img, scores_img, nms_iou_threshold)
        
        final_boxes = boxes_img[keep_indices]
        final_scores = scores_img[keep_indices]
        final_labels = labels_img[keep_indices]

        output_batch.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels + 1 
        })
    return output_batch


@torch.no_grad()
def evaluate_detection(model, val_loader, device, writer, epoch, stage_name):
    model.eval()
    map_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device) # Keep True for detailed map per class if needed
    
    for images_from_loader, targets_list in tqdm(val_loader, desc=f"Eval Det Epoch {epoch+1}", leave=False):
        if not isinstance(images_from_loader, (tuple, list)):
            images = images_from_loader.to(device)
        elif not all(isinstance(img_tensor, torch.Tensor) for img_tensor in images_from_loader):
             raise TypeError(f"Expected images_from_loader to be a sequence of Tensors, but found other types within.")
        else:
            images = torch.stack(images_from_loader, dim=0).to(device)
            
        det_outputs_raw, _, _ = model(images)
        preds_formatted = decode_detection_outputs(det_outputs_raw, model) 

        targets_formatted = []
        for t_dict in targets_list:
            formatted_target_boxes = t_dict['boxes'].to(device)
            formatted_target_labels = t_dict['labels'].to(device)
            
            valid_boxes_mask = (formatted_target_boxes[:, 2] > formatted_target_boxes[:, 0]) & \
                               (formatted_target_boxes[:, 3] > formatted_target_boxes[:, 1])
            
            targets_formatted.append({
                'boxes': formatted_target_boxes[valid_boxes_mask],
                'labels': formatted_target_labels[valid_boxes_mask]
            })
            
        if any(p['boxes'].numel() > 0 for p in preds_formatted) or any(t['boxes'].numel() > 0 for t in targets_formatted):
             map_metric.update(preds_formatted, targets_formatted)
    
    try:
        map_results = map_metric.compute()
        mAP = map_results['map'].item()
        mAP50 = map_results['map_50'].item()
    except Exception as e:
        print(f"Error computing mAP: {e}. Returning dummy mAP.")
        mAP, mAP50 = 0.0, 0.0
    map_metric.reset()

    print(f"Epoch {epoch+1} Det Val: mAP: {mAP:.4f}, mAP@0.5: {mAP50:.4f}")
    if writer:
        writer.add_scalar(f"{stage_name}/Det_Val_mAP", mAP, epoch)
        writer.add_scalar(f"{stage_name}/Det_Val_mAP50", mAP50, epoch)
    return mAP


@torch.no_grad()
def evaluate_classification(model, val_loader, device, writer, epoch, stage_name):
    model.eval()
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSIFICATION_CLASSES).to(device)
    total_loss = 0
    for images, labels in tqdm(val_loader, desc=f"Eval Cls Epoch {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)
        _, _, cls_outputs = model(images)
        loss = cls_criterion(cls_outputs, labels)
        total_loss += loss.item()
        accuracy_metric.update(cls_outputs, labels)
        
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_metric.compute().item()
    accuracy_metric.reset()
    
    if writer:
        writer.add_scalar(f"{stage_name}/Cls_Val_Loss", avg_loss, epoch)
        writer.add_scalar(f"{stage_name}/Cls_Val_Accuracy", acc, epoch)
    print(f"Epoch {epoch+1} Cls Val: Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return acc

# --- Training Functions for each stage ---
def train_one_epoch(model, teacher_model, train_loader, optimizer, device, epoch, writer,
                    task_name, current_stage_name,
                    lambda_lwf_seg_val=0.0, lambda_lwf_det_val=0.0, current_task_idx=0,
                    # Replay Buffer 參數
                    replay_seg_loader=None, replay_lambda_seg_val=0.0,
                    replay_det_loader=None, replay_lambda_det_val=0.0):
    model.train()
    if teacher_model:
        teacher_model.eval()

    # 初始化 Replay Buffer 的 iterators
    # 這樣可以確保在一個 epoch 內，如果 replay data 耗盡，可以從頭開始
    local_iter_replay_seg = iter(replay_seg_loader) if replay_seg_loader else None
    local_iter_replay_det = iter(replay_det_loader) if replay_det_loader else None
    
    total_loss_epoch = 0.0
    total_task_loss_epoch = 0.0
    total_scaled_lwf_loss_epoch = 0.0
    total_scaled_replay_loss_epoch = 0.0 # 新增：記錄 replay loss

    progress_bar = tqdm(train_loader, desc=f"Train {task_name} E{epoch+1}", leave=False)

    for batch_idx, (images, targets) in enumerate(progress_bar):
        
        if task_name == "Detection":
            images_stacked = torch.stack(images, dim=0).to(device)
            # targets for detection 是 list of dicts，由 detection_criterion 內部處理 device
        else:  
            images_stacked = images.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        det_outputs, seg_outputs, cls_outputs = model(images_stacked)

        # --- 主任務損失 ---
        task_loss = torch.tensor(0.0, device=device)
        det_loc_loss_item, det_conf_loss_item, det_cls_loss_item = 0,0,0 # 用於記錄
        if task_name == "Segmentation":
            task_loss = seg_criterion(seg_outputs, targets)
        elif task_name == "Detection":
            # detection_criterion 返回 (total_loss, loc_loss, conf_loss, cls_loss)
            # 這些都已經是 batch 內的總和
            task_loss, det_loc_loss_item, det_conf_loss_item, det_cls_loss_item = detection_criterion(det_outputs, targets)
            if writer and batch_idx % 50 == 0: # Log sub-losses
                writer.add_scalar(f"{current_stage_name}/Det_Train_Loc_Loss_Batch", det_loc_loss_item.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f"{current_stage_name}/Det_Train_Conf_Loss_Batch", det_conf_loss_item.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f"{current_stage_name}/Det_Train_Cls_Loss_Batch", det_cls_loss_item.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)
        elif task_name == "Classification":
            task_loss = cls_criterion(cls_outputs, targets)
        
        # --- LwF 蒸餾損失 ---
        scaled_lwf_loss_components_sum = torch.tensor(0.0, device=device)
        if teacher_model:
            with torch.no_grad():
                teacher_det_out, teacher_seg_out, teacher_cls_out = teacher_model(images_stacked)

            if current_task_idx >= 1 and lambda_lwf_seg_val > 0: # Stage 2 (Det) or Stage 3 (Cls)
                lwf_s = F.mse_loss(seg_outputs, teacher_seg_out) # 蒸餾分割 logits
                scaled_lwf_loss_components_sum += lambda_lwf_seg_val * lwf_s
            
            if current_task_idx >= 2 and lambda_lwf_det_val > 0: # Stage 3 (Cls)
                # 保持對檢測原始輸出的蒸餾，或考慮更細緻的蒸餾策略
                lwf_d = F.mse_loss(det_outputs, teacher_det_out) 
                scaled_lwf_loss_components_sum += lambda_lwf_det_val * lwf_d
        
        # --- Replay Buffer 損失 ---
        scaled_replay_loss_component = torch.tensor(0.0, device=device)
        if current_task_idx == 1 and local_iter_replay_seg and replay_lambda_seg_val > 0: # 訓練偵測時，回放分割
            try:
                replay_s_images, replay_s_targets = next(local_iter_replay_seg)
            except StopIteration: # 如果 replay data 耗盡，從頭開始
                local_iter_replay_seg = iter(replay_seg_loader) 
                replay_s_images, replay_s_targets = next(local_iter_replay_seg)
            
            replay_s_images, replay_s_targets = replay_s_images.to(device), replay_s_targets.to(device)
            _, seg_outputs_replay, _ = model(replay_s_images) # 用當前模型預測
            loss_replay_s = seg_criterion(seg_outputs_replay, replay_s_targets)
            scaled_replay_loss_component += replay_lambda_seg_val * loss_replay_s

        elif current_task_idx == 2: # 訓練分類時，回放分割和偵測
            if local_iter_replay_seg and replay_lambda_seg_val > 0:
                try:
                    replay_s_images, replay_s_targets = next(local_iter_replay_seg)
                except StopIteration:
                    local_iter_replay_seg = iter(replay_seg_loader)
                    replay_s_images, replay_s_targets = next(local_iter_replay_seg)
                replay_s_images, replay_s_targets = replay_s_images.to(device), replay_s_targets.to(device)
                _, seg_outputs_replay, _ = model(replay_s_images)
                loss_replay_s = seg_criterion(seg_outputs_replay, replay_s_targets)
                scaled_replay_loss_component += replay_lambda_seg_val * loss_replay_s

            if local_iter_replay_det and replay_lambda_det_val > 0:
                try:
                    replay_d_images_tuple, replay_d_targets_list = next(local_iter_replay_det)
                except StopIteration:
                    local_iter_replay_det = iter(replay_det_loader)
                    replay_d_images_tuple, replay_d_targets_list = next(local_iter_replay_det)
                
                replay_d_images_stacked = torch.stack(replay_d_images_tuple, dim=0).to(device)
                det_outputs_replay, _, _ = model(replay_d_images_stacked)
                # detection_criterion 返回 (total_loss, loc_loss, conf_loss, cls_loss)
                loss_replay_d_total, _, _, _ = detection_criterion(det_outputs_replay, replay_d_targets_list)
                scaled_replay_loss_component += replay_lambda_det_val * loss_replay_d_total
        
        # 總損失 = 主任務損失 + LwF損失 + Replay損失
        # 這些損失都應該是 batch 內的總和
        total_loss = task_loss + scaled_lwf_loss_components_sum + scaled_replay_loss_component
        
        # 標準化總損失 (除以主任務的 batch size)
        # LwF 和 Replay 的損失是基於它們各自的 batch size (主任務 batch size 或 replay batch size)
        # 這裡的 total_loss 是一個 scalar sum.
        # 如果 task_loss, lwf_loss, replay_loss 都是各自 batch 內的和，
        # 那麼 total_loss.backward() 是合理的。
        # Optimizer step 會基於這個總和的梯度。
        # 為了日誌記錄的一致性，通常會將 batch loss 除以 batch size
        # 這裡的 task_loss, lwf_loss, replay_loss 都是 sum over their respective batches.
        # 如果我們想讓 learning rate 的影響更可預測，也許應該將每個 loss component 除以它自己的 batch size
        # 但目前 LwF 是對 images_stacked 做的，所以 batch size 和主任務一樣。
        # Replay loss 的 batch size 是 replay_batch_size。
        # 為了簡化，我們先假設權重 lambda 已經隱含了 batch size 的差異。

        total_loss_normalized_for_backward = total_loss / images_stacked.size(0) # 除以主任務 batch size

        total_loss_normalized_for_backward.backward()
        optimizer.step()
        
        # 記錄損失時，也用 normalized loss
        total_loss_epoch += total_loss_normalized_for_backward.item()
        total_task_loss_epoch += task_loss.item() / images_stacked.size(0) # 主任務損失標準化
        total_scaled_lwf_loss_epoch += scaled_lwf_loss_components_sum.item() / images_stacked.size(0) # LwF 標準化
        if scaled_replay_loss_component.item() != 0 : # 只有當 replay loss 存在時才累加和標準化
             # Replay loss 的 batch size 可能不同，但 lambda 應已調節
             # 為了日誌，可以除以其自身的 batch size (如果能獲取到) 或保持原樣由 lambda 控制
             # 這裡我們除以主任務的 batch size 以保持一致性，lambda 應反映這一點
            total_scaled_replay_loss_epoch += scaled_replay_loss_component.item() / images_stacked.size(0) 
        
        progress_bar.set_postfix({
            'Total L (norm)': f'{total_loss_normalized_for_backward.item():.4f}',
            'Task L': f'{task_loss.item() / images_stacked.size(0):.4f}', # 顯示標準化後的
            'LwF L': f'{scaled_lwf_loss_components_sum.item() / images_stacked.size(0):.4f}',
            'Replay L': f'{scaled_replay_loss_component.item() / images_stacked.size(0) if images_stacked.size(0) > 0 else 0:.4f}'
        })
        
        if writer and batch_idx % 50 == 0:
             writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Total_Loss_Norm_Batch", total_loss_normalized_for_backward.item(), epoch * len(train_loader) + batch_idx)
             writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Task_Loss_Norm_Batch", task_loss.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)
             if scaled_lwf_loss_components_sum.item() > 0:
                writer.add_scalar(f"{current_stage_name}/{task_name}_Train_LwF_Loss_Scaled_Norm_Batch", scaled_lwf_loss_components_sum.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)
             if scaled_replay_loss_component.item() > 0: 
                writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Replay_Loss_Scaled_Norm_Batch", scaled_replay_loss_component.item() / images_stacked.size(0), epoch * len(train_loader) + batch_idx)

    avg_total_loss = total_loss_epoch / len(train_loader)
    avg_task_loss = total_task_loss_epoch / len(train_loader)
    avg_scaled_lwf_loss = total_scaled_lwf_loss_epoch / len(train_loader)
    avg_scaled_replay_loss = total_scaled_replay_loss_epoch / len(train_loader) 

    if writer:
        writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Total_Loss_Norm_Epoch", avg_total_loss, epoch)
        writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Task_Loss_Norm_Epoch", avg_task_loss, epoch)
        if avg_scaled_lwf_loss > 0 : # 僅在 LwF 啟用時記錄
             writer.add_scalar(f"{current_stage_name}/{task_name}_Train_LwF_Loss_Scaled_Norm_Epoch", avg_scaled_lwf_loss, epoch)
        if avg_scaled_replay_loss > 0 : # 僅在 Replay 啟用時記錄
             writer.add_scalar(f"{current_stage_name}/{task_name}_Train_Replay_Loss_Scaled_Norm_Epoch", avg_scaled_replay_loss, epoch)
    print(f"Epoch {epoch+1} {task_name} Train: Avg Total Loss (Norm): {avg_total_loss:.4f}, Avg Task Loss (Norm): {avg_task_loss:.4f}, Avg LwF Loss (Norm): {avg_scaled_lwf_loss:.4f}, Avg Replay Loss (Norm): {avg_scaled_replay_loss:.4f}")
    return avg_total_loss


# --- Main Function ---
def main(ARGS):
    set_seed(ARGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and ARGS.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    if ARGS.num_anchors > len(ANCHOR_BOXES):
        print(f"Error: ARGS.num_anchors ({ARGS.num_anchors}) cannot be greater than the number of defined ANCHOR_BOXES ({len(ANCHOR_BOXES)}).")
        return
    active_anchors = ANCHOR_BOXES[:ARGS.num_anchors]
    print(f"Using {ARGS.num_anchors} anchors: {active_anchors}")

    log_dir = os.path.join(ARGS.log_dir, f"{ARGS.experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs will be saved to: {log_dir}")

    model = UnifiedMultiTaskNet(
        num_det_classes=NUM_DETECTION_CLASSES,
        num_seg_classes=NUM_SEGMENTATION_CLASSES,
        num_cls_classes=NUM_CLASSIFICATION_CLASSES,
        num_anchors_det=ARGS.num_anchors
    ).to(device)
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    teacher_model = None # 用於 LwF

    # --- 初始化資料集 ---
    seg_img_transform_train, seg_mask_transform_train = get_segmentation_transforms(train=True)
    seg_img_transform_val, seg_mask_transform_val = get_segmentation_transforms(train=False)
    seg_train_dataset = VOCSegmentationDataset(root_dir=os.path.join(ARGS.data_root, "mini_voc_seg"), split='train', img_transform=seg_img_transform_train, mask_transform=seg_mask_transform_train, train_phase=True)
    seg_val_dataset = VOCSegmentationDataset(root_dir=os.path.join(ARGS.data_root, "mini_voc_seg"), split='val', img_transform=seg_img_transform_val, mask_transform=seg_mask_transform_val, train_phase=False)
    
    det_train_dataset = COCODetectionDataset(root_dir=os.path.join(ARGS.data_root, "mini_coco_det"), ann_file_name="instances_train_mini.json", split='train')
    det_val_dataset = COCODetectionDataset(root_dir=os.path.join(ARGS.data_root, "mini_coco_det"), ann_file_name="instances_val_mini.json", split='val')

    cls_train_transform = get_transform(train=True, task="classification")
    cls_val_transform = get_transform(train=False, task="classification")
    cls_train_dataset = ImagenetteClassificationDataset(root_dir=os.path.join(ARGS.data_root, "imagenette_160"), split='train', transform=cls_train_transform)
    cls_val_dataset = ImagenetteClassificationDataset(root_dir=os.path.join(ARGS.data_root, "imagenette_160"), split='val', transform=cls_val_transform)

    seg_train_loader = DataLoader(seg_train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.num_workers, pin_memory=True)
    seg_val_loader = DataLoader(seg_val_dataset, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.num_workers, pin_memory=True)
    det_train_loader = DataLoader(det_train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.num_workers, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    det_val_loader = DataLoader(det_val_dataset, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.num_workers, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    cls_train_loader = DataLoader(cls_train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.num_workers, pin_memory=True)
    cls_val_loader = DataLoader(cls_val_dataset, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.num_workers, pin_memory=True)

    metrics_history = {
        "mIoU_base": 0.0, "mAP_base": 0.0, "Top1_base": 0.0,
        "mIoU_final": 0.0, "mAP_final": 0.0, "Top1_final": 0.0,
    }
    best_mIoU_stage1 = 0.0
    best_mAP_stage2 = 0.0
    best_Top1_stage3 = 0.0
    
    global detection_criterion # 更新 detection_criterion 的實例
    detection_criterion = DetectionLoss(num_classes=NUM_DETECTION_CLASSES,
                                        anchors_config=active_anchors,
                                        feature_map_stride=IMG_HEIGHT // (IMG_HEIGHT // 32) 
                                       ).to(device)
    
    # --- Replay Buffer DataLoaders ---
    replay_seg_loader_main = None
    replay_det_loader_main = None

    # --- Stage 1: Train Segmentation ---
    print("\n--- Starting Stage 1: Segmentation Training (Task Index 0) ---")
    optimizer_s1 = get_optimizer(model, ARGS.lr_seg, ARGS.optimizer)
    scheduler_s1 = get_scheduler(optimizer_s1, ARGS.scheduler, ARGS.epochs_seg)
    
    for epoch in range(ARGS.epochs_seg):
        train_one_epoch(model, None, seg_train_loader, optimizer_s1, device, epoch, writer,
                        "Segmentation", "Stage1_Seg", current_task_idx=0) # Replay 和 LwF 在此階段不啟用
        current_mIoU = evaluate_segmentation(model, seg_val_loader, device, writer, epoch, "Stage1_Seg")
        if scheduler_s1: scheduler_s1.step()
        if current_mIoU > best_mIoU_stage1:
            best_mIoU_stage1 = current_mIoU
            # torch.save(model.state_dict(), os.path.join(ARGS.checkpoint_dir, f"{ARGS.experiment_name}_stage1_best.pth"))
    metrics_history["mIoU_base"] = best_mIoU_stage1
    writer.add_scalar("ForgetMetrics/mIoU_base", best_mIoU_stage1, 1) # Stage 1 結束時的 global_step 設為 1
    print(f"Stage 1 (Seg) Best mIoU (mIoU_base): {best_mIoU_stage1:.4f}")
    
    teacher_model = copy.deepcopy(model).to(device) # 準備 LwF 的教師模型
    teacher_model.eval()

    if ARGS.use_replay_buffer and ARGS.replay_lambda_seg > 0:
        print("Creating replay buffer for Segmentation task...")
        num_replay_samples_s = min(len(seg_train_dataset), ARGS.replay_buffer_size)
        replay_seg_indices = random.sample(range(len(seg_train_dataset)), k=num_replay_samples_s)
        replay_seg_subset = Subset(seg_train_dataset, replay_seg_indices)
        replay_seg_loader_main = DataLoader(
            replay_seg_subset, batch_size=ARGS.replay_batch_size, 
            shuffle=True, num_workers=ARGS.num_workers, pin_memory=True, drop_last=True # drop_last 確保 iterator 行為一致
        )
        print(f"Replay buffer for Segmentation created with {num_replay_samples_s} samples, batch size {ARGS.replay_batch_size}.")


    # --- Stage 2: Train Detection ---
    print("\n--- Starting Stage 2: Detection Training (Task Index 1) ---")
    optimizer_s2 = get_optimizer(model, ARGS.lr_det, ARGS.optimizer)
    scheduler_s2 = get_scheduler(optimizer_s2, ARGS.scheduler, ARGS.epochs_det)

    for epoch in range(ARGS.epochs_det):
        train_one_epoch(model, teacher_model, det_train_loader, optimizer_s2, device, epoch, writer,
                        "Detection", "Stage2_Det",
                        lambda_lwf_seg_val=ARGS.lambda_lwf_seg, current_task_idx=1,
                        replay_seg_loader=replay_seg_loader_main if ARGS.use_replay_buffer else None, 
                        replay_lambda_seg_val=ARGS.replay_lambda_seg if ARGS.use_replay_buffer else 0.0)
        current_mAP = evaluate_detection(model, det_val_loader, device, writer, epoch, "Stage2_Det")
        if scheduler_s2: scheduler_s2.step()
        if current_mAP > best_mAP_stage2:
            best_mAP_stage2 = current_mAP
            # torch.save(model.state_dict(), os.path.join(ARGS.checkpoint_dir, f"{ARGS.experiment_name}_stage2_best.pth"))
    metrics_history["mAP_base"] = best_mAP_stage2
    writer.add_scalar("ForgetMetrics/mAP_base", best_mAP_stage2, 2) # Stage 2 結束時的 global_step 設為 2
    print(f"Stage 2 (Det) Best mAP (mAP_base): {best_mAP_stage2:.4f}")

    mIoU_after_s2 = evaluate_segmentation(model, seg_val_loader, device, writer, ARGS.epochs_det, "Post_Stage2_SegEval")
    mIoU_drop_s2_val = metrics_history["mIoU_base"] - mIoU_after_s2
    writer.add_scalar("ForgetMetrics/mIoU_drop_after_Stage2_absolute", mIoU_drop_s2_val, 2)
    print(f"Seg mIoU after Stage 2 (Det training): {mIoU_after_s2:.4f}, Absolute Drop from base: {mIoU_drop_s2_val:.4f}")
    
    teacher_model = copy.deepcopy(model).to(device) # 更新 LwF 的教師模型
    teacher_model.eval()

    if ARGS.use_replay_buffer and ARGS.replay_lambda_det > 0:
        print("Creating replay buffer for Detection task...")
        num_replay_samples_d = min(len(det_train_dataset), ARGS.replay_buffer_size)
        replay_det_indices = random.sample(range(len(det_train_dataset)), k=num_replay_samples_d)
        replay_det_subset = Subset(det_train_dataset, replay_det_indices)
        replay_det_loader_main = DataLoader(
            replay_det_subset, batch_size=ARGS.replay_batch_size,
            shuffle=True, num_workers=ARGS.num_workers, pin_memory=True, drop_last=True,
            collate_fn=lambda x: tuple(zip(*x)) 
        )
        print(f"Replay buffer for Detection created with {num_replay_samples_d} samples, batch size {ARGS.replay_batch_size}.")


    # --- Stage 3: Train Classification ---
    print("\n--- Starting Stage 3: Classification Training (Task Index 2) ---")
    optimizer_s3 = get_optimizer(model, ARGS.lr_cls, ARGS.optimizer)
    scheduler_s3 = get_scheduler(optimizer_s3, ARGS.scheduler, ARGS.epochs_cls)

    for epoch in range(ARGS.epochs_cls):
        train_one_epoch(model, teacher_model, cls_train_loader, optimizer_s3, device, epoch, writer,
                        "Classification", "Stage3_Cls",
                        lambda_lwf_seg_val=ARGS.lambda_lwf_seg, 
                        lambda_lwf_det_val=ARGS.lambda_lwf_det, current_task_idx=2,
                        replay_seg_loader=replay_seg_loader_main if ARGS.use_replay_buffer else None, 
                        replay_lambda_seg_val=ARGS.replay_lambda_seg if ARGS.use_replay_buffer else 0.0,
                        replay_det_loader=replay_det_loader_main if ARGS.use_replay_buffer else None,
                        replay_lambda_det_val=ARGS.replay_lambda_det if ARGS.use_replay_buffer else 0.0)
        current_top1 = evaluate_classification(model, cls_val_loader, device, writer, epoch, "Stage3_Cls")
        if scheduler_s3: scheduler_s3.step()
        if current_top1 > best_Top1_stage3:
            best_Top1_stage3 = current_top1
            # torch.save(model.state_dict(), os.path.join(ARGS.checkpoint_dir, f"{ARGS.experiment_name}_stage3_best.pth"))
    metrics_history["Top1_base"] = best_Top1_stage3 
    metrics_history["Top1_final"] = best_Top1_stage3 
    writer.add_scalar("ForgetMetrics/Top1_final", best_Top1_stage3, 3) # Stage 3 結束時的 global_step 設為 3
    print(f"Stage 3 (Cls) Best Top-1 Accuracy (Top1_final): {best_Top1_stage3:.4f}")

    # --- Final Evaluation ---
    metrics_history["mIoU_final"] = evaluate_segmentation(model, seg_val_loader, device, writer, ARGS.epochs_cls, "Final_SegEval")
    metrics_history["mAP_final"] = evaluate_detection(model, det_val_loader, device, writer, ARGS.epochs_cls, "Final_DetEval")
    final_top1_at_end = evaluate_classification(model, cls_val_loader, device, writer, ARGS.epochs_cls, "Final_ClsEval_Check") 
    
    mIoU_drop_final_percent = 0.0
    if metrics_history["mIoU_base"] > 1e-6:
        mIoU_drop_final_percent = (metrics_history["mIoU_base"] - metrics_history["mIoU_final"]) / metrics_history["mIoU_base"] * 100
    
    mAP_drop_final_percent = 0.0
    if metrics_history["mAP_base"] > 1e-6:
        mAP_drop_final_percent = (metrics_history["mAP_base"] - metrics_history["mAP_final"]) / metrics_history["mAP_base"] * 100
    
    # Top-1 檢查的是絕對下降值是否超過 0.05 (5%)
    top1_abs_drop = metrics_history["Top1_base"] - final_top1_at_end # Top1_base 是 Stage 3 期間的最佳值
    
    writer.add_scalar("ForgetMetrics/mIoU_final", metrics_history["mIoU_final"], 3)
    writer.add_scalar("ForgetMetrics/mAP_final", metrics_history["mAP_final"], 3)
    writer.add_scalar("ForgetMetrics/Top1_at_very_end", final_top1_at_end, 3)
    writer.add_scalar("ForgetMetrics/mIoU_drop_final_percent", mIoU_drop_final_percent, 3)
    writer.add_scalar("ForgetMetrics/mAP_drop_final_percent", mAP_drop_final_percent, 3)
    writer.add_scalar("ForgetMetrics/Top1_abs_drop_from_peak", top1_abs_drop, 3)

    print("\n--- Final Forgetting Check ---")
    print(f"mIoU Base (End of Stage 1): {metrics_history['mIoU_base']:.4f}")
    print(f"mIoU Final (End of Stage 3): {metrics_history['mIoU_final']:.4f}")
    print(f"mIoU Drop: {mIoU_drop_final_percent:.2f}% (Target Drop <= 5%)")

    print(f"mAP Base (End of Stage 2): {metrics_history['mAP_base']:.4f}")
    print(f"mAP Final (End of Stage 3): {metrics_history['mAP_final']:.4f}")
    print(f"mAP Drop: {mAP_drop_final_percent:.2f}% (Target Drop <= 5%)")

    print(f"Top-1 Base (Best during Stage 3): {metrics_history['Top1_base']:.4f}")
    print(f"Top-1 at very end: {final_top1_at_end:.4f}")
    print(f"Top-1 Absolute Drop from Base: {top1_abs_drop:.4f} (Target Drop <= 0.05)") # 0.05 代表 5%
    
    pass_criteria = True
    if mIoU_drop_final_percent > 5.0:
        print(f"FAIL: mIoU dropped by {mIoU_drop_final_percent:.2f}% (more than 5%)")
        pass_criteria = False
    if mAP_drop_final_percent > 5.0:
        print(f"FAIL: mAP dropped by {mAP_drop_final_percent:.2f}% (more than 5%)")
        pass_criteria = False
    if top1_abs_drop > 0.05: 
         print(f"FAIL: Top-1 dropped by {top1_abs_drop:.4f} (more than 0.05 absolute) from its peak of {metrics_history['Top1_base']:.4f}")
         pass_criteria = False

    if pass_criteria:
        print("\nAll tasks are within their respective performance drop criteria!")
    else:
        print("\nOne or more tasks EXCEEDED their performance drop criteria.")

    # 保存最終模型
    final_model_path = os.path.join(ARGS.checkpoint_dir, f"{ARGS.experiment_name}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()
    print("Training finished. Tensorboard logs saved.")
    print(f"To view logs: tensorboard --logdir={ARGS.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified One-Head Multi-Task Training with LwF and Replay Buffer")
    parser.add_argument('--data_root', type=str, default='./data', help="Root directory of datasets")
    parser.add_argument('--log_dir', type=str, default='./logs_lwf_replay', help="Directory for Tensorboard logs") # 更新 log dir
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_lwf_replay', help="Directory for model checkpoints") # 更新 checkpoint dir
    parser.add_argument('--experiment_name', type=str, default='multitask_lwf_replay_v1', help="Name for this experiment run")
    
    parser.add_argument('--lr_seg', type=float, default=1e-3, help="Learning rate for segmentation stage")
    parser.add_argument('--lr_det', type=float, default=1e-4, help="Learning rate for detection stage")
    parser.add_argument('--lr_cls', type=float, default=1e-3, help="Learning rate for classification stage")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for main task training")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers")
    
    parser.add_argument('--epochs_seg', type=int, default=5, help="Epochs for segmentation training (可減少以加速調試)")
    parser.add_argument('--epochs_det', type=int, default=5, help="Epochs for detection training (可減少以加速調試)")
    parser.add_argument('--epochs_cls', type=int, default=5, help="Epochs for classification training (可減少以加速調試)")

    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help="Optimizer type")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'], help="LR scheduler type")
    
    parser.add_argument('--num_anchors', type=int, default=3, choices=range(1, len(ANCHOR_BOXES) + 1), help="Number of anchors per location for detection head.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--use_cuda', type=bool, default=True, help="Use CUDA if available")
    
    # LwF 參數
    parser.add_argument('--lambda_lwf_seg', type=float, default=1.0, help="Strength of LwF for segmentation task when training subsequent tasks.")
    parser.add_argument('--lambda_lwf_det', type=float, default=0.5, help="Strength of LwF for detection task when training classification.")

    # Replay Buffer 參數
    parser.add_argument('--use_replay_buffer', action='store_true', help="Enable Replay Buffer strategy.")
    parser.add_argument('--replay_batch_size', type=int, default=5, help="Batch size for replay loaders (should be <= 10 per task).")
    parser.add_argument('--replay_buffer_size', type=int, default=100, help="Number of samples to keep in memory for each replay task.")
    parser.add_argument('--replay_lambda_seg', type=float, default=0.5, help="Strength of Replay for segmentation task.")
    parser.add_argument('--replay_lambda_det', type=float, default=0.5, help="Strength of Replay for detection task.")
    # parser.add_argument('--reinit_replay_per_epoch', action='store_true', help="Reinitialize replay iterators at the start of each main task epoch.") # 這個參數暫時不加，iterator 會在耗盡時自動重新初始化

    ARGS = parser.parse_args()

    os.makedirs(ARGS.log_dir, exist_ok=True)
    os.makedirs(ARGS.checkpoint_dir, exist_ok=True)

    main(ARGS)