import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
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

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        self.stage1 = nn.Sequential(*features[:23])  # 到pool3
        self.stage2 = nn.Sequential(*features[23:33])  # 到pool4
        self.stage3 = nn.Sequential(*features[33:43])  # 到pool5

        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool5 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)
    def forward(self, x):
        pool3 = self.stage1(x)
        pool4 = self.stage2(pool3)
        pool5 = self.stage3(pool4)

        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)
        score_pool5 = self.score_pool5(pool5)

        upscore2 = self.upscore2(score_pool5)
        fuse_pool4 = upscore2 + score_pool4

        upscore4 = self.upscore4(fuse_pool4)
        fuse_pool3 = upscore4 + score_pool3

        out = self.upscore8(fuse_pool3)
        # 新增：调整输出尺寸与输入一致
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

def get_voc_loaders(batch_size=4, img_size=256, root='./data'):
    # 单独的归一化步骤，只应用于图像
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    # 定义图像和mask的联合变换
    class JointTransform:
        def __init__(self, img_size):
            self.img_size = img_size
            # 是否进行水平翻转的随机状态
            self.flip = False
            # 随机裁剪的参数
            self.i, self.j, self.h, self.w = 0, 0, img_size, img_size
            
        def __call__(self, image, mask):
            # 调整大小
            image = transforms.functional.resize(image, (self.img_size + 32, self.img_size + 32))
            mask = transforms.functional.resize(mask, (self.img_size + 32, self.img_size + 32), 
                                              interpolation=transforms.InterpolationMode.NEAREST)
            
            # 随机裁剪 - 只在第一次调用时确定裁剪参数
            if self.h == self.img_size:  # 用这个作为标记判断是否第一次调用
                self.i, self.j, self.h, self.w = transforms.RandomCrop.get_params(
                    image, output_size=(self.img_size, self.img_size))
                
            # 对图像和mask应用相同的裁剪
            image = transforms.functional.crop(image, self.i, self.j, self.h, self.w)
            mask = transforms.functional.crop(mask, self.i, self.j, self.h, self.w)
            
            # 随机水平翻转 - 只在第一次调用时决定是否翻转
            if not hasattr(self, 'flip_decided'):
                self.flip = torch.rand(1) < 0.5
                self.flip_decided = True
                
            # 对图像和mask应用相同的翻转
            if self.flip:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
                
            # 图像特有变换：转tensor和归一化
            image = transforms.functional.to_tensor(image)
            image = normalize(image)
            
            # Mask变换：转numpy然后转tensor
            mask = torch.as_tensor(np.array(mask).copy())
            
            return image, mask
    
    # 创建自定义VOC数据集类以使用联合变换
    class VOCWithJointTransform(VOCSegmentation):
        def __init__(self, root, year, image_set, transform, download=False):
            super().__init__(root=root, year=year, image_set=image_set, download=download,
                            transform=None, target_transform=None)
            self.joint_transform = transform
            
        def __getitem__(self, index):
            img = Image.open(self.images[index]).convert('RGB')
            mask = Image.open(self.masks[index])
            
            # 应用联合变换
            if self.joint_transform is not None:
                # 每个样本创建新的变换实例，确保随机性
                transform = JointTransform(img_size)
                img, mask = transform(img, mask)
                
            return img, mask.long()
    
    # 使用新的联合变换创建数据集
    train_set = VOCWithJointTransform(root=root, year='2012', image_set='train', 
                                    transform=JointTransform(img_size), download=False)
    val_set = VOCWithJointTransform(root=root, year='2012', image_set='val',
                                  transform=JointTransform(img_size), download=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    epoch_str = str(epoch)
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch_str}, Batch {batch_idx+1}/{len(loader)} Loss: {loss.item():.4f}")
    return total_loss / len(loader)

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
            loss = criterion(outputs, masks.squeeze(1))
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu()
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

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_predictions(model, loader, device, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu()
    for i in range(imgs.size(0)):
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        axes[1].imshow(masks[i].numpy())
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(preds[i].numpy())
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        plt.savefig(f'{save_dir}/comparison_{i}.png')
        plt.close(fig)

import os
import logging
from datetime import datetime

if __name__ == "__main__":
    # 创建带日期格式的文件夹
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'results_{current_time}'
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志记录
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 21
    model = FCN8s(num_classes=num_classes).to(device)
    train_loader, val_loader = get_voc_loaders(batch_size=32, img_size=256, root='../DeepLearn2/data')
    weights = torch.ones(num_classes)
    weights[0] = 0.1  # 背景类别权重为0.1，其余为1
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    start_time = time.time()

    for epoch in range(200):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, miou = validate(model, val_loader, criterion, device, epoch, num_classes)
        scheduler.step()  # 每个epoch结束后调用
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, mIoU={miou:.4f}")

    final_val_loss, final_miou = validate(model, val_loader, criterion, device, epoch='final', num_classes=num_classes)
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}, Final mIoU: {final_miou:.4f}")
    save_predictions(model, val_loader, device, save_dir=save_dir)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Training Time: {total_time:.2f} seconds")