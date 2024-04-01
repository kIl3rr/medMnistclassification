import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt 

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

model = tf.keras.models.load_model('./model/medical_mnist_model.keras')
image = cv2.imread('./MedMNIST/CXR/000001.jpeg',cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(128,128))
image = image / 255.0
image = np.expand_dims(image, axis=-1)

prediction = model.predict(np.array([image]))

a = np.argmax(prediction[0])

plt.subplot(1,1,1)
plt.imshow(image)

name = {
    0: "腹部",
    1: "乳房",
    2: "肺部",
    3: "胸部",
    4: "头部"
}

# print()
plt.title(name[a])
plt.show()