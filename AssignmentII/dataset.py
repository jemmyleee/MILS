# dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import random

# --- Configuration ---
# Define expected number of classes for each task (for consistency, can be passed to model)
NUM_DETECTION_CLASSES = 10  # Mini-COCO has 10 classes (re-mapped 1-10)
NUM_SEGMENTATION_CLASSES = 21 # PASCAL VOC has 20 classes + 1 background
NUM_CLASSIFICATION_CLASSES = 10 # Imagenette has 10 classes

# Standard image size for input to the model
IMG_WIDTH = 512 # As per assignment inference speed test
IMG_HEIGHT = 512

# --- Transformations ---
def get_transform(train=True, task="classification"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_list = []
    if train:
        transform_list.append(transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.5, 1.0)))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Resize((IMG_HEIGHT, IMG_WIDTH)))

    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)
    
    return transforms.Compose(transform_list)

def get_segmentation_transforms(train=True):
    img_transform_list = []
    if train:
        img_transform_list.append(transforms.Resize((IMG_HEIGHT, IMG_WIDTH))) 
    else:
        img_transform_list.append(transforms.Resize((IMG_HEIGHT, IMG_WIDTH)))
    
    img_transform_list.append(transforms.ToTensor())
    img_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    img_transforms = transforms.Compose(img_transform_list)

    mask_transform_list = []
    mask_transform_list.append(transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST))
    mask_transform_list.append(transforms.ToTensor()) 
    
    mask_transforms = transforms.Compose(mask_transform_list)
    
    return img_transforms, mask_transforms


# --- Dataset Classes ---

class COCODetectionDataset(Dataset):
    def __init__(self, root_dir, ann_file_name, split='train', transform=None):
        self.img_dir = os.path.join(root_dir, split)
        self.ann_path = os.path.join(root_dir, 'annotations', ann_file_name)
        self.coco = COCO(self.ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # self.transform = transform # transform is applied internally now for simplicity

        ids_with_anns = set()
        for ann_id in self.coco.anns:
            ids_with_anns.add(self.coco.anns[ann_id]['image_id'])
        self.ids = [img_id for img_id in self.ids if img_id in ids_with_anns]
        
        # Basic transform for detection images
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        original_w, original_h = img.size

        boxes = []
        labels = []
        
        for ann in coco_anns:
            if ann.get('ignore', 0) == 1 or ann.get('iscrowd', 0) == 1: # Ignore crowd annotations for simplicity
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: # Skip invalid boxes
                continue
            boxes.append([x, y, x + w, y + h]) # COCO format: x_min, y_min, width, height -> x_min, y_min, x_max, y_max
            labels.append(ann['category_id'])

        if not boxes: # If no valid annotations, return a dummy target or skip (here we'll create dummy)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id]) # Keep for evaluation if needed
        target["original_size"] = torch.as_tensor([int(original_h), int(original_w)]) # For scaling boxes back if needed

        img = self.img_transform(img)
            
        # Scale boxes to the new image size (IMG_HEIGHT, IMG_WIDTH)
        if target["boxes"].shape[0] > 0:
            new_h, new_w = IMG_HEIGHT, IMG_WIDTH
            boxes = target["boxes"]
            boxes[:, 0] = boxes[:, 0] * (new_w / original_w) # x_min
            boxes[:, 1] = boxes[:, 1] * (new_h / original_h) # y_min
            boxes[:, 2] = boxes[:, 2] * (new_w / original_w) # x_max
            boxes[:, 3] = boxes[:, 3] * (new_h / original_h) # y_max
            # Clip boxes to image dimensions
            boxes[:, 0::2].clamp_(min=0, max=new_w)
            boxes[:, 1::2].clamp_(min=0, max=new_h)
            target["boxes"] = boxes

        return img, target

    def __len__(self):
        return len(self.ids)

class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', img_transform=None, mask_transform=None, train_phase=True):
        self.img_dir = os.path.join(root_dir, split)
        self.mask_dir = os.path.join(root_dir, split) 
        self.train_phase = train_phase

        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        img_name = self.images[index]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # Usually L mode (grayscale) or P mode (palette)

        if self.train_phase and random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            
        if self.img_transform:
            img = self.img_transform(img)
        
        if self.mask_transform:
            mask = self.mask_transform(mask) 
            mask = (mask * 255).squeeze(0).long() 
            # PASCAL VOC: 0-20 classes, 255 for border/void.
            # The CrossEntropyLoss ignore_index handles 255.
        return img, mask

    def __len__(self):
        return len(self.images)

class ImagenetteClassificationDataset(ImageFolder):
    def __init__(self, root_dir, split='train', transform=None):
        data_path = os.path.join(root_dir, split)
        super().__init__(root=data_path, transform=transform)

