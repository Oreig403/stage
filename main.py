from matplotlib import pyplot as plt
import tensorflow.keras as keras

import numpy as np
from keras import layers, models, datasets
from keras.datasets import mnist

def show_img(arr, vmin = 0, vmax = 255, filename = None):
    plt.imshow(arr, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    if filename:
        plt.savefig(filename)
    plt.show()



(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
show_img(X_train[200], 0, 255, "example.png")

X_train = X_train/255
X_test = X_test / 255

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

model = models.Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(60, activation="relu"),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')

