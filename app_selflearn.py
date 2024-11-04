from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_cors import CORS 
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import base64
from keras.preprocessing import image
import threading
import shutil

# Initialize Flask application
app = Flask(__name__)
CORS(app)
app.secret_key = 'ilovemaven'  
data_dir = "segmented"
feedback_dir = "feedback"
os.makedirs(feedback_dir, exist_ok=True)

# Load the model
model = load_model("pflanz_arzt_v4.h5")

image_size = (150, 150)
class_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_class_labels(segmented_dir='segmented'):
    class_labels = [d for d in os.listdir(segmented_dir) if os.path.isdir(os.path.join(segmented_dir, d))]
    return class_labels

def collect_feedback(image_path, predicted_label, correct_label):

     # Copy the misclassified image to the feedback folder
    feedback_image_path = os.path.join(feedback_dir, os.path.basename(image_path))
    shutil.copy(image_path, feedback_image_path) 

    #Transfer to training folder
    data_image_path = os.path.join(data_dir, correct_label, os.path.basename(image_path))
    os.rename(image_path, data_image_path)

    # Save the feedback to a CSV file for future retraining
    with open("feedback_labels.csv", "a") as f:
        f.write(f"{feedback_image_path};{correct_label}\n")  # Save path and correct label

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    # Save the uploaded image
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # Preprocess the image for prediction
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions, axis=1)[0]]
    certainty = np.max(predictions)

    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # Render feedback.html with the data
    return render_template('feedback.html',
                           image_path=base64_image,
                           predicted_class=predicted_class,
                           certainty=certainty,
                           class_labels=class_labels,
                           full_path=image_path)

@app.route('/feedback', methods=['POST'])
def feedback():
    image_path = request.form['image_path']
    predicted_class = request.form['predicted_class']
    correct_label = request.form['correct_label']

    if correct_label:
        collect_feedback(image_path, predicted_class, correct_label)

        #update_model() -> update only on request

    return redirect(url_for('index'))

@app.route('/update-model', methods=['GET'])
def update_model():
    # Load feedback data
    if os.path.exists("feedback_labels.csv"):
        feedback_data = pd.read_csv("feedback_labels.csv", sep=';', header=None, names=['filename', 'class'])
        new_image_paths = feedback_data['filename'].values
        new_labels = feedback_data['class'].values

        # Preprocess the new images
        X_new = []
        for img_path in new_image_paths:
            img_array = preprocess_image(img_path)
            X_new.append(img_array)

        # Convert to a single numpy array
        X_new = np.vstack(X_new)  # Shape will be (num_new_images, 150, 150, 1)
        y_new = np.array(new_labels)

        # Load existing training data
        X_train, y_train = load_data(data_dir)
        
        # Concatenate new and existing data
        y_train_encoded = to_categorical(y_train, num_classes=len(class_labels))
        y_new_encoded = to_categorical(y_new, num_classes=len(class_labels))

        # Concatenate new and existing data
        all_image_arrays = np.concatenate((X_train, X_new))
        all_labels = np.concatenate((y_train_encoded, y_new_encoded))

        # Retrain the model
        model.fit(all_image_paths, all_labels, epochs=3)  
        model.save("pflanz_arzt_v4_a.h5")
        print("Model retrained with new feedback data.")

def load_data(data_dir):
    image_arrays = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')): 
                    img_path = os.path.join(class_dir, img_file)
                    img_array = preprocess_image(img_path)  # Preprocess the image
                    image_arrays.append(img_array)  
                    labels.append(label)
    return np.vstack(image_arrays), np.array(labels)  # Returns a 4D array of images

if __name__ == '__main__':
    app.run(debug=True)
