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