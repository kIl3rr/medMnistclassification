import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 添加 Dropout 到分类器中
        vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.vgg = vgg
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.vgg(x)
        x = self.dropout(x)
        return x

# 实例化模型
model = ClassificationModel(num_classes=10)  # 假设有10个类别
model.eval()  # 将模型设置为评估模式

# 创建一个兼容的输入张量
x = torch.randn(1, 3, 224, 224)  # 输入应与模型期望的输入大小相匹配

# 导出模型到 ONNX
torch.onnx.export(model, x, "classification_model.onnx", export_params=True, opset_version=10,
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
