from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import base64
from keras.preprocessing import image
import shutil

from segment import segment_image

app = Flask(__name__)
CORS(app)
app.secret_key = 'ilovemaven'

data_dir = "segmented"
feedback_dir = "feedback"
os.makedirs(feedback_dir, exist_ok=True)

model = load_model("pflanz_arzt_v5-2.h5")

image_size = (150, 150)
class_labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

best_predict = ""
best_cert = 0 

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def collect_feedback(image_path, predicted_label, correct_label):
    feedback_image_path = os.path.join(feedback_dir, os.path.basename(image_path))
    shutil.copy(image_path, feedback_image_path)

    data_image_path = os.path.join(data_dir, correct_label, os.path.basename(image_path))
    os.rename(image_path, data_image_path)

    with open("feedback_labels.csv", "a") as f:
        f.write(f"{feedback_image_path};{correct_label}\n")

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

    segmented_output_dir = os.path.join("segmented_output", os.path.splitext(file.filename)[0])
    os.makedirs(segmented_output_dir, exist_ok=True)

    try:
        segmented_images = segment_image(img_path, segmented_output_dir)  # List of segmented image paths
    except Exception as e:
        return jsonify({'error': f"Segmentation failed: {str(e)}"}), 500

    best_prediction = None
    highest_certainty = 0.0

    for segmented_image in segmented_images:
        try:
            img_array = preprocess_image(segmented_image)

            predictions = model.predict(img_array)
            certainty = np.max(predictions)  # Highest confidence
            predicted_class = class_labels[np.argmax(predictions, axis=1)[0]]
            print(predicted_class)
            print(certainty)

            if certainty > highest_certainty:
                highest_certainty = certainty
                best_prediction = predicted_class
                best_cert = certainty
                best_predict = predicted_class

        except Exception as e:
            print(f"[ERROR] Failed to process {segmented_image}: {str(e)}")
    
    with open(img_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({'predicted_class': best_prediction})


@app.route('/feedback', methods=['POST'])
def feedback():
    image_path = request.form['image_path']
    predicted_class = request.form['predicted_class']
    correct_label = request.form['correct_label']

    if correct_label:
        collect_feedback(image_path, predicted_class, correct_label)

    return redirect(url_for('index'))

@app.route('/update-model', methods=['GET'])
def update_model():
    if os.path.exists("feedback_labels.csv"):
        feedback_data = pd.read_csv("feedback_labels.csv", sep=';', header=None, names=['filename', 'class'])
        new_image_paths = feedback_data['filename'].values
        new_labels = feedback_data['class'].values

        X_new = []
        for img_path in new_image_paths:
            img_array = preprocess_image(img_path)
            X_new.append(img_array)

        X_new = np.vstack(X_new)
        y_new = np.array(new_labels)

        X_train, y_train = load_data(data_dir)
        y_train_encoded = to_categorical(y_train, num_classes=len(class_labels))
        y_new_encoded = to_categorical(y_new, num_classes=len(class_labels))

        all_image_arrays = np.concatenate((X_train, X_new))
        all_labels = np.concatenate((y_train_encoded, y_new_encoded))

        model.fit(all_image_arrays, all_labels, epochs=3)
        model.save("pflanz_arzt_v4_a.h5")
        print("Model retrained with new feedback data.")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # preprosses the image for prediction
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions, axis=1)[0]]
    certainty = np.max(predictions)

    print(best_predict)
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template('feedback.html',
                           image_path=base64_image,
                           predicted_class=best_predict,
                           certainty=best_cert,
                           class_labels=class_labels,
                           full_path=image_path)

def load_data(data_dir):
    image_arrays = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    img_array = preprocess_image(img_path)
                    image_arrays.append(img_array)
                    labels.append(label)
    return np.vstack(image_arrays), np.array(labels)

if __name__ == '__main__':
    app.run(debug=True)
