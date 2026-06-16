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


X_train = X_train/255
X_test = X_test / 255

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

model = models.Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(90, activation="relu"),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')

# Get predictions on test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Store incorrect predictions
incorrect_samples = []
for i in range(len(Y_test)):
    if predicted_classes[i] != Y_test[i]:
        incorrect_samples.append({
            'image': X_test[i],
            'correct_label': Y_test[i],
            'predicted_label': predicted_classes[i]
        })


#print("correct label :", incorrect_samples[100]['correct_label'])
#print("predicted label :", incorrect_samples[100]['predicted_label'])
#print(f'Number of incorrect predictions: {len(incorrect_samples)}')
#show_img(incorrect_samples[100]['image'], 0, 1, "incorrect.png")
print(len(X_test)+ len(X_train))

