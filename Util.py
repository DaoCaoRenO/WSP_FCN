import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision
from torchvision.datasets import VOCSegmentation
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch.nn.functional as F
from PIL import Image
import os
import logging
from datetime import datetime

def logging_setup(log_dir='logs'):
      # 设置日志文件名，带时间戳
    log_filename = f"log/train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    # 用logging.info替换print
    print = logging.info
    return print

def compute_miou(preds, masks, num_classes, ignore_index=255):
    preds = preds.numpy()
    masks = masks.numpy()
    # print(torch.unique(preds), torch.unique(masks))

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (masks == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # 忽略该类
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def validate(model, loader, criterion, device, epoch, num_classes=21, save_dir='results'):
    model.eval()
    total_loss = 0
    epoch_str = str(epoch)
    all_preds = []
    all_masks = []
    
    # 创建保存当前epoch验证结果的目录
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch_str}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    # 用于保存可视化的图像、掩码和预测
    vis_imgs = []
    vis_masks = []
    vis_preds = []
    
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            masks = masks.to(device).long()
            outputs = model(imgs)
            
            # 判断模型类型并提取正确的输出
            if isinstance(model, torchvision.models.segmentation.DeepLabV3):
                output = outputs['out']  # 提取主要输出
                loss = criterion(output, masks.squeeze(1))
            else:
                output = outputs  # 直接使用输出
                loss = criterion(output, masks.squeeze(1))
            
            total_loss += loss.item()
            
            # 计算预测结果
            preds = torch.argmax(output, dim=1).cpu()  # 使用主要输出
            print("preds unique:", torch.unique(preds))
            all_preds.append(preds)
            all_masks.append(masks.cpu().squeeze(1))
            
            # 保存第一批次的图像用于可视化（假设批量大小>=10）
            if batch_idx == 2 and len(vis_imgs) < 10:
                # 取前10张或批量大小（取较小值）
                num_to_save = min(10, imgs.size(0))
                for i in range(num_to_save):
                    vis_imgs.append(imgs[i].cpu())
                    vis_masks.append(masks[i].cpu().squeeze(0))  # 去掉通道维度
               
                    vis_preds.append(preds[i])
            
            print(f"Epoch {epoch_str}, Validation Batch {batch_idx+1}/{len(loader)} Loss: {loss.item():.4f}")
    
    # 计算平均损失和mIoU
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    miou = compute_miou(all_preds, all_masks, num_classes)
    print(f"Epoch {epoch_str}, Validation Loss: {avg_loss:.4f}, mIoU: {miou:.4f}")
    
    # 保存可视化结果
    if vis_imgs:
        for i in range(len(vis_imgs)):
            # 反标准化图像
            img_np = vis_imgs[i].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            # 创建颜色映射，使背景（0类）为黑色
            cmap = plt.cm.get_cmap('tab20', num_classes)
            colors = cmap(np.arange(num_classes))
            colors[0] = [0, 0, 0, 1]  # 背景设为黑色
            custom_cmap = plt.cm.colors.ListedColormap(colors)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(vis_masks[i].numpy(), cmap=custom_cmap, vmin=0, vmax=num_classes-1)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(vis_preds[i].numpy(), cmap=custom_cmap, vmin=0, vmax=num_classes-1)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{epoch_save_dir}/comparison_{i}.png')
            plt.close(fig)
        
        print(f"Saved {len(vis_imgs)} validation image comparisons to {epoch_save_dir}")
    
    return avg_loss, miou