import os
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# 定义分类模型，使用VGG16
class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        vgg = vgg16(pretrained=True)
        # 添加 Dropout 到分类器中
        vgg.classifier[6] = nn.Linear(4096,num_classes)
        self.vgg = vgg
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.vgg(x)
        x = self.dropout(x)
        return x

# 数据转换定义
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 数据集路径
data_dir = 'C:\\Users\\Administrator\\Desktop\\project\\MedMNIST'
dataset = datasets.ImageFolder(data_dir, transform=transform)

# 将数据集索引分为训练集和测试集
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

# 创建训练集和测试集的SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=2)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler, num_workers=2)

# 初始化模型、损失函数和优化器
model = ClassificationModel(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train_and_save_model(model, scheduler, train_loader, criterion, optimizer, num_epochs=10, device='cuda', save_path='vgg16age_classification_model.pth'):
    model.to(device)
    best_val_loss = float('inf')
    log_txt = open("vgg_log.txt","a")  # 在训练开始前打开日志文件
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': train_loss / len(progress_bar)})
        scheduler.step()  # 在每个 epoch 结束后更新学习率
        epoch_train_loss = train_loss / len(train_loader)
        log_txt.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}\n')
        if epoch_train_loss < best_val_loss:
            best_val_loss = epoch_train_loss
            torch.save(model.state_dict(), save_path)
        progress_bar.close()
    print(f"Training complete. Best training loss: {best_val_loss:.4f}. Model saved to '{save_path}'.")
    log_txt.close()

# 定义评估函数
def evaluate(model, data_loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    log_txt = open("vgg_log.txt","a")
    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    log_txt.write(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    log_txt.close()

# 使用
if __name__ == "__main__":
    device = torch.device("cuda")
    train_and_save_model(model, scheduler, train_loader, criterion, optimizer, num_epochs=10, device=device, save_path='vgg16age_classification_model.pth')
    model.load_state_dict(torch.load('vgg16age_classification_model.pth'))
    model.to(device)
    evaluate(model, test_loader, criterion, device=device)
