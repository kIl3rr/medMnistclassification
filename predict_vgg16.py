def classification_from_img():
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import vgg16
    import torch.nn as nn
    import tkinter as tk
    from tkinter import filedialog

    class AgeEstimationModel(nn.Module):
        def __init__(self, num_classes=6):
            super(AgeEstimationModel, self).__init__()
            vgg = vgg16(pretrained=True)
            # 添加 Dropout 到分类器中
            vgg.classifier[6] = nn.Linear(4096,num_classes)
            self.vgg = vgg
            self.dropout = nn.Dropout(0.5)
        def forward(self,x):
            x = self.vgg(x)
            x = self.dropout(x)
            return x
    def select_image():
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename()  # Open file dialog and return the selected file path
        return file_path

    # Create an instance of the age estimation model
    model = AgeEstimationModel()

    # Load the trained model states
    age_model_path = "C:\\Users\\Administrator\\Desktop\\project\\model\\vgg16_classification_model.pth"
    state_dict = torch.load(age_model_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)

    # Set models to evaluation mode
    model.eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor()
    ])

    # Select an image using a file dialog
    image_path = select_image()
    if not image_path:
        print("No image selected.")
        exit()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a dimension to simulate batch_size of 1

    # 使用模型进行预测
    with torch.no_grad():
        output = model(image)

    # 获取预测结果（例如获取分类结果）
    predicted_class_index = torch.argmax(output,dim=1).item()
    result_list = ["腹部CT","乳房MRI","胸部","胸部X线","手","头部CT"]
    return result_list[predicted_class_index]

x = classification_from_img()
print(x)