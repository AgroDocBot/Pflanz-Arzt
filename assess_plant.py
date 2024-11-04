import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import subprocess
import os

# Load the pre-trained model
model = tf.keras.models.load_model('pflanz_arzt_v3_pcv.h5')

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict_image_class(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    return predictions

def get_class_labels(segmented_dir='segmented'):
    class_labels = [d for d in os.listdir(segmented_dir) if os.path.isdir(os.path.join(segmented_dir, d))]
    return class_labels

def print_prediction(predictions, class_labels):
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted Class: {class_labels[predicted_class]}")
    print(f"Prediction Probabilities: {predictions}")

# Test the model on an image
img_path = 'testimg.jpeg'
predictions = predict_image_class(model, img_path)
class_labels = get_class_labels()  # Fetch class labels from the segmented directory
print_prediction(predictions, class_labels)
