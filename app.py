from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model('pflanz_arzt_v3_pcv.h5')

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_class_labels(segmented_dir='segmented'):
    class_labels = [d for d in os.listdir(segmented_dir) if os.path.isdir(os.path.join(segmented_dir, d))]
    return class_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_path = 'uploads/' + file.filename
    file.save(img_path)

    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_labels = get_class_labels()
    predicted_class = class_labels[np.argmax(predictions, axis=1)[0]]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
