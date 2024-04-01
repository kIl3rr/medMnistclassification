from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('../model/medical_mnist_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # 保存上传的图片    
            image_path = './uploads/' + file.filename
            file.save(image_path)

            # 进行预测
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            image = image / 255.0
            image = np.expand_dims(image, axis=-1)

            prediction = model.predict(np.array([image]))
            class_index = np.argmax(prediction[0])

            name = {
                0: "腹部",
                1: "乳房",
                2: "肺部",
                3: "胸部",
                4: "头部"
            }

            return render_template('result.html', image=image_path, prediction=name[class_index])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file("uploads/"+filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
