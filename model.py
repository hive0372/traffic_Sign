import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Reshape, MaxPooling2D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
from PIL import Image

# Assign data paths
train = 'Train/'
test = 'Test/'
categories = len(os.listdir(train))
print('Categories:', categories)

# Load data
data = []
labels = []
height, width, channels = 32, 32, 3

for i in range(categories):
    path = os.path.join(train, str(i))
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(os.path.join(path, img))
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((height, width))
            data.append(np.array(resize_image))
            labels.append(i)
        except:
            print("Error in loading image")

data = np.array(data)
labels = np.array(labels)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)
x_train = x_train / 255.0
x_val = x_val / 255.0

# One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, categories)
y_val_cat = keras.utils.to_categorical(y_val, categories)

# Define the CNN model
model = Sequential([
    InputLayer(input_shape=(height, width, channels)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(categories, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Custom callback to track precision, recall, and F1-score per epoch
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.precision = []
        self.recall = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = model.predict(x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val_cat, axis=1)

        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)

        print(f"Epoch {epoch+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

metrics_callback = MetricsCallback()

# Train the model with the custom callback
history = model.fit(x_train, y_train_cat, batch_size=64, epochs=5, validation_data=(x_val, y_val_cat), callbacks=[metrics_callback])

# Evaluate the model
model.save("RetrievaNet43_model.h5")

# Plot Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Precision
plt.figure(figsize=(6, 4))
plt.plot(metrics_callback.precision, color='green', marker='o', linestyle='--', label='Validation Precision')
plt.title('Precision Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# Plot Recall
plt.figure(figsize=(6, 4))
plt.plot(metrics_callback.recall, color='purple', marker='o', linestyle='--', label='Validation Recall')
plt.title('Recall Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()

# Plot F1-Score
plt.figure(figsize=(6, 4))
plt.plot(metrics_callback.f1, color='orange', marker='o', linestyle='--', label='Validation F1-Score')
plt.title('F1-Score Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.show()
