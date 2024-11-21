import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt 

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

model = tf.keras.models.load_model('./model/medical_mnist_model.keras')
image = cv2.imread('./MedMNIST/HeadCT/000001.jpeg',cv2.IMREAD_GRAYSCALE) #使用OpenCV的imread函数读取指定路径下的JPEG图像文件，以灰度模式读取。
image = cv2.resize(image,(128,128))
image = image / 255.0
image = np.expand_dims(image, axis=-1) #在图像的最后一个维度上增加一个维度，将图像从二维变为三维，满足模型输入的需求（模型通常期望输入是多个通道，即使是灰度图像也需要显式地表示通道数）。

prediction = model.predict(np.array([image]))

a = np.argmax(prediction[0])

plt.subplot(1,1,1)
plt.imshow(image)

name = {
    0: "腹部",
    1: "乳房",
    2: "肺部",
    3: "胸部",
    4: "手",
    5: "头部"
}

print(a)
# plt.title(name[a])
# plt.show()