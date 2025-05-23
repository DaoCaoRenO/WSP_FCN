from datetime import datetime
import os
import time
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import Util
import DataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# 检查设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # 加载 VOC 数据集
# train_dataset = torchvision.datasets.VOCSegmentation(
#     root="./data", year="2012", image_set="train", download=True, transform=transform, target_transform=transform
# )
# val_dataset = torchvision.datasets.VOCSegmentation(
#     root="./data", year="2012", image_set="val", download=True, transform=transform, target_transform=transform
# )

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

train_loader, val_loader = DataSet.get_voc_loaders(batch_size=8, img_size=256, root='../DeepLearn2/data')

# 加载 DeepLabv3 模型
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 21, kernel_size=1)  # VOC 有 21 个类别
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=255)  # 忽略标注为 255 的像素
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
# 训练和验证函数
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
       
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, targets.long())
            running_loss += loss.item()
    return running_loss / len(dataloader)

# 训练循环
num_epochs = 200
best_loss = float("inf")
save_path =  f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print=Util.logging_setup(save_path)
# os.makedirs(save_path, exist_ok=True)

total_start = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)   
    val_loss, val_miou = Util.validate(model, val_loader, criterion, device, epoch, num_classes=21, save_dir=save_path)
    epoch_time = time.time() - epoch_start
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
    print(f"Epoch {epoch+1} time: {epoch_time:.2f} seconds")
     # 学习率调度器放在这里
    scheduler.step(val_loss)

    # # 保存最优模型
    # if val_loss < best_loss:
    #     best_loss = val_loss
    #     torch.save(model.state_dict(), "best_deeplabv3_voc.pth")
    #     print("Model saved!")

total_time = time.time() - total_start
print(f"Training complete! Total time: {total_time:.2f} seconds")