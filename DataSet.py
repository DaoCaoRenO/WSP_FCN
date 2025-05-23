from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np


def get_voc_loaders(batch_size=4, img_size=256, root='../DeepLearn2/data'):
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
        # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            image = transforms.functional.crop(image, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

            # 随机水平翻转
            if torch.rand(1) < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            # 随机垂直翻转
            if torch.rand(1) < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            # 随机旋转
            angle = float(torch.empty(1).uniform_(-10, 10))
            image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = transforms.functional.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # 颜色抖动
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = color_jitter(image)

            # 转tensor和归一化
            image = transforms.functional.to_tensor(image)
            image = normalize(image)
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