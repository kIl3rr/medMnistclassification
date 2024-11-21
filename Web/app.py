import random
import shutil
import threading
from uuid import uuid4
from flask import Flask, jsonify, render_template, request, send_file
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vgg16
import torch.nn as nn
from torchvision.models import resnet101
from torchvision.models import densenet121

app = Flask(__name__)
class_names = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
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

class vgg16model(nn.Module):
    def __init__(self, num_classes=6):
        super(vgg16model, self).__init__()
        vgg = vgg16(pretrained=True)
        # 添加 Dropout 到分类器中
        vgg.classifier[6] = nn.Linear(4096,num_classes)
        self.vgg = vgg
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.vgg(x)
        x = self.dropout(x)
        return x

class ResNet101ClassificationModel(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet101ClassificationModel, self).__init__()
        self.resnet = resnet101(pretrained=True)  # 使用ResNet101
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)  # 在全连接层之前使用 Dropout
        return x

class DenseNet121ClassificationModel(nn.Module):
    def __init__(self, num_classes=6):
        super(DenseNet121ClassificationModel, self).__init__()
        densenet = densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 在全连接层之前使用 Dropout
        x = self.classifier(x)
        return x

@app.route('/')
def index():
    return render_template('index.html')

task={}
def list_files(directory):
    # 使用os.scandir()迭代目录中的所有条目
    with os.scandir(directory) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
    return files


@app.route('/res',methods=['POST'])
def res():

    model_choice = request.form['model']  # 获取模型选择
    task_id = str(uuid4())
    task[task_id] = {"status": "Processing", "results": {}}
    def process_task():
        # 设置文件夹路径
        directory_path = 'C:\\Users\\Administrator\\Desktop\\project\\test'

        # 获取文件夹中的文件列表
        files = list_files(directory_path)

        # 打印文件列表
        for file in files:
            try:
                predict(file,model_choice,task_id)
            except Exception as e:
                print(e)
        task[task_id]["status"] = "Complete"    
    threading.Thread(target=process_task).start()    
    return jsonify({"task_id":task_id})

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    task_status = task.get(task_id, {})
    task[task_id] = {"status": "Processing", "results": {}}
    return jsonify(task_status)

def predict(file,model_choice,task_id):
    
    image_path = '../test/' + file

    # 根据选择的模型进行预测
    if model_choice == 'model1':
        model1 = tf.keras.models.load_model('../model/medical_mnist_model.keras')
        # 现有的 TensorFlow 模型预测逻辑
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        prediction = model1.predict(np.array([image]))
        class_index = np.argmax(prediction[0])
        name = {
            0: "腹部",
            1: "乳房",
            2: "胸部",
            3: "CXR",
            4: "手",
            5: "头部"
        }
        n2 = {
            0: "AbdomenCT",
            1: "BreastMRI",
            2: "ChestCT",
            3: "CXR",
            4: "Hand",
            5: "HeadCT"
        }
        task[task_id]["results"][file]=name[class_index]
        new_path = os.path.join('./results/' + str(n2[class_index]), file)
        # Move file
        move_file(image_path, new_path)
        # return render_template('result.html', image=image_path, prediction=name[class_index])
    elif model_choice == 'model2':

        model2 = ConvolutionalNetwork()
        model2 = torch.load('../model/convolutional_network_full.pth')

        model2.eval()

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
        
        image_tensor = load_image(image_path)

        # Move the image tensor to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        model2.to(device)

        with torch.no_grad():
            output = model2(image_tensor)
            prediction = torch.argmax(output, dim=1)
            predicted_class = class_names[prediction.item()]

        name = {
            "AbdomenCT": "腹部",
            "BreastMRI": "乳房",
            "ChestCT": "胸部",
            "CXR": "CXR",
            "HeadCT": "头部",
            "Hand": "手"
        }
        task[task_id]["results"][file]=name[predicted_class]
        new_path = os.path.join('./results/' + str(predicted_class), file)
        # Move file
        move_file(image_path, new_path)
        # return render_template('result.html', image=image_path, prediction=name[predicted_class])
    elif model_choice == 'model3':
        model3 = vgg16model()
        state_dict = torch.load('../model/vgg16_classification_model.pth', map_location=torch.device('cuda'))
        model3.load_state_dict(state_dict)
        model3.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor()
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add a dimension to simulate batch_size of 1

        # 使用模型进行预测
        with torch.no_grad():
            output = model3(image)

        # 获取预测结果（例如获取分类结果）
        predicted_class_index = torch.argmax(output,dim=1).item()
        result_list = ["腹部","乳房","胸部","CXR","手","头部"]

        r2 = ["AbdomenCT","BreastMRI","ChestCT","CXR","Hand","HeadCT"]
        task[task_id]["results"][file]=result_list[predicted_class_index]

        new_path = os.path.join('./results/' + str(r2[predicted_class_index]), file)
        # Move file
        move_file(image_path, new_path)
        # return render_template('result.html', image=image_path,prediction=result_list[predicted_class_index])
    elif model_choice == 'model4':
        model4 = ResNet101ClassificationModel()
        state_dict = torch.load('../model/resnet_classification_model.pth', map_location=torch.device('cuda'))
        model4.load_state_dict(state_dict)
        model4.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor()
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add a dimension to simulate batch_size of 1

        # 使用模型进行预测
        with torch.no_grad():
            output = model4(image)

        # 获取预测结果（例如获取分类结果）
        predicted_class_index = torch.argmax(output,dim=1).item()
        result_list = ["腹部","乳房","胸部","CXR","手","头部"]
        r2 = ["AbdomenCT","BreastMRI","ChestCT","CXR","Hand","HeadCT"]
        task[task_id]["results"][file]=result_list[predicted_class_index]
        new_path = os.path.join('./results/' + str(r2[predicted_class_index]), file)
        # Move file
        move_file(image_path, new_path)
        # return render_template('result.html', image=image_path,prediction=result_list[predicted_class_index])
    elif model_choice == 'model5':
        model5 = DenseNet121ClassificationModel()
        state_dict = torch.load('../model/densenet121_classification_model.pth', map_location=torch.device('cuda'))
        model5.load_state_dict(state_dict)

        # Set models to evaluation mode
        model5.eval()

        # Image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor()
        ])
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add a dimension to simulate batch_size of 1

        # 使用模型进行预测
        with torch.no_grad():
            output = model5(image)

        # 获取预测结果（例如获取分类结果）
        predicted_class_index = torch.argmax(output,dim=1).item()
        result_list = ["腹部","乳房","胸部","CXR","手","头部"]
        r2 = ["AbdomenCT","BreastMRI","ChestCT","CXR","Hand","HeadCT"]
        task[task_id]["results"][file]=result_list[predicted_class_index]
        new_path = os.path.join('./results/' + str(r2[predicted_class_index]), file)
        # Move file
        move_file(image_path, new_path)
        # return render_template('result.html', image=image_path,prediction=result_list[predicted_class_index])
        
def move_file(src, dst):
    try:
        os.rename(src, dst)
        return "File moved successfully."
    except FileNotFoundError:
        return "Source file does not exist."
    except OSError as e:
        return f"Error moving file: {e}"      

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):



#     return send_file("uploads/"+filename, mimetype='image/jpeg')
@app.route('/clean')
def clear_folder():
    task_id = str(uuid4())
    task[task_id] = {"status": "Processing", "results": {}}
    def process_task():
        folder_paths = ['C:\\Users\Administrator\\Desktop\\project\\test','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\AbdomenCT','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\BreastMRI','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\ChestCT','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\CXR','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\Hand','C:\\Users\Administrator\\Desktop\\project\\Web\\results\\HeadCT']
        for folder_path in folder_paths:
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                print(folder_path)
                print("文件夹不存在")
                return

            # 遍历文件夹中的所有文件和子文件夹
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子文件夹
                except Exception as e:
                    print(f'删除 {file_path} 失败. 原因: {e}')
        
            task[task_id]["status"] = "Complete"    
    threading.Thread(target=process_task).start()    
    return jsonify({"task_id":task_id})

@app.route('/pre')
def prepareTestdata():
    task_id = str(uuid4())
    task[task_id] = {"status": "Processing", "results": {}}
    def process_task():
        # 源文件夹路径列表
        source_directories = [
            '../MedMNIST/1',
            '../MedMNIST/2',
            '../MedMNIST/3',
            '../MedMNIST/4',
            '../MedMNIST/5',
            '../MedMNIST/6'
        ]

        # 目标文件夹路径
        destination_directory = '../test'

        # 确保目标文件夹存在
        os.makedirs(destination_directory, exist_ok=True)

        # 遍历每个源文件夹
        for folder in source_directories:
            # 获取文件夹内所有文件
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpeg')]
            
            # 随机选择文件数量，例如1到1000
            number_of_files_to_copy = random.randint(1, 30)
            
            # 随机选择指定数量的文件
            selected_files = random.sample(files, number_of_files_to_copy)
            
            # 复制选定的文件到目标文件夹
            for file in selected_files:
                # 创建目标文件路径
                destination_path = os.path.join(destination_directory, os.path.basename(file))
                # 复制文件
                shutil.copy(file, destination_path)
                print(f'Copied {file} to {destination_path}')
        task[task_id]["status"] = "Complete" 
    threading.Thread(target=process_task).start()    
    return jsonify({"task_id":task_id})
if __name__ == '__main__':
    print('current work directory: '+os.getcwd())
    app.run(debug=True)