import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from create_model import create_model

# Directory path for all processed images
data_dir = "segmented"

# Parameters
epochs = 3
batch_size = 32
learning_rate = 0.001

image_paths = []
labels = []

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):  # Adjust extensions as needed
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.02, stratify=labels, random_state=42
)

# Parameters for number of classes
num_classes = len(np.unique(train_labels))  # Detects number of classes

print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_paths)}")
print(f"Testing samples: {len(test_paths)}")

# Model
model = create_model(num_classes)  
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_paths, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

model.save("pflanz_arzt_v3_pcv.h5")
print("Model training complete and saved as pflanz_arzt_v3.h5")

loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy:.4f}")
