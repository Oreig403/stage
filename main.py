from matplotlib import pyplot as plt
import tensorflow.keras as keras

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, models



# Charger le dataset dans la mémoire
digits = load_digits()
X = digits.images  # Shape: (1797, 8, 8)
Y = digits.target  # Shape: (1797,)
X = X / 16.0 
Y = to_categorical(Y, num_classes=10)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

# Définition du modele
model = models.Sequential([
    layers.Flatten(input_shape = (8,8)),
    layers.Dense(50, activation="relu"),
    layers.Dense(30, activation="relu"),
    layers.Dense(10, activation='softmax')
])

#Entrainer le modèle 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')