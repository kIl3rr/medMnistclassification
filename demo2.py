import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
import os

dir0 = './MedMNIST'
names0 = os.listdir(dir0)
names = sorted(names0)
# print(names)

train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=dir0, transform=train_transform)


class_names = dataset.classes
print(class_names)

#定义了一个卷积神经网络模型，包括卷积层、池化层和全连接层。
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

model = ConvolutionalNetwork()
model = torch.load('./model/convolutional_network_full.pth')

model.eval()

import torch
from torchvision import transforms
from PIL import Image

# Assuming the model and class_names are already defined and loaded as per your previous code

# Function to load an image and transform it
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB in case it's not
    transformation = transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Load and transform the image
image_path = './MedMNIST/HeadCT/000001.jpeg'  # Update the path to your specific image
image_tensor = load_image(image_path)

# Move the image tensor to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
model.to(device)

# Predict the class
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
    predicted_class = class_names[prediction.item()]


# predicted_class
print("Predicted Class:", predicted_class)

