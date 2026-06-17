from matplotlib import pyplot as plt
from math import * 
import tensorflow.keras as keras

import numpy as np
from keras import layers, models, datasets
from keras.datasets import mnist

def show_img(arr, vmin = 0, vmax = 255, filename = None):
    plt.imshow(arr, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    if filename:
        plt.savefig(filename)
    plt.show()

def show_img_with_value(arr, vmin = None, vmax = None, filename = None):
    arr = np.asarray(arr)
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()

    fig, ax = plt.subplots()
    im = ax.imshow(arr, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    threshold = (vmin + vmax) / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if isinstance(value, float):
                text = f"{value:.2f}"
            else:
                text = str(value)
            text_color = 'white' if value < threshold else 'black'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=7)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()


def logMat(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            if (m[i][j] > 0):
                m[i][j] = log10(m[i][j])

    return m


def confusion_matrix(predicted_values, test_values):
    m = [[0 for j in range(10)] for i in range(10)]
    #for (pred_label, correct_label) in zip(predicted_values, test_values):
    for i in range(len(test_values)):
        m[test_values[i]][predicted_values[i]] += 1
    
    for i in range(10):
        row_sum = sum(m[i])
        if row_sum == 0:
            continue
        for j in range(10):
            m[i][j] = (m[i][j] / row_sum) * 100

    return m


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()


X_train = X_train / 255
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

m = confusion_matrix(predicted_classes, Y_test)

print(m)
show_img_with_value(m, 0, 100, "confusion.png")

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

