from matplotlib import pyplot as plt
import tensorflow.keras as keras

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, models

def show_img(arr, vmin = 0, vmax = 255):
    plt.imshow(arr, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

# 1. Load the dataset
digits = load_digits()
X = digits.images  # Shape: (1797, 8, 8)
Y = digits.target  # Shape: (1797,)

X = X / 16.0 

# 3. One-hot encode the labels for Keras
Y = to_categorical(Y, num_classes=10)

# 4. Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

model = models.Sequential([
    layers.Flatten(input_shape = (8,8)),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')