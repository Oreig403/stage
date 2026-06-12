from matplotlib import pyplot as plt
import tensorflow.keras as keras

import numpy as np
from keras import layers, models, datasets
from keras.datasets import mnist

def show_img(arr, vmin = 0, vmax = 255):
    plt.imshow(arr, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(60, activation="relu"),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, Y_test)


y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

Y_test_labels = np.argmax(Y_test, axis=1)

failed_indices = np.where(y_pred_labels != Y_test_labels)[0]

print(f"Failed samples: {len(failed_indices)}")

for idx in failed_indices[:10]:
    print(
        f"Index={idx}, "
        f"True={Y_test[idx]}, "
        f"Pred={y_pred_labels[idx]}, "
        f"Prob={y_pred[idx][0]:.4f}"
    )
    show_img(X_test[idx])
print(f'Test accuracy: {test_acc}')