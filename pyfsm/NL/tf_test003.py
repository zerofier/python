import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset = load_digits()
image_shape = (8, 8, 1)
num_class = 10
y = dataset.target
y = tf.keras.utils.to_categorical(y, num_class)
X = dataset.data
X = np.array([data.reshape(image_shape) for data in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        5, kernel_size=3, strides=1, padding='same',
        input_shape=image_shape, activation='relu'),
    tf.keras.layers.Conv2D(
        5, kernel_size=3, strides=1, padding='same',
        activation='relu'),
    tf.keras.layers.Conv2D(
        3, kernel_size=2, strides=1, padding='same',
        activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=num_class, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(X_train, y_train, epochs=8)

predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
actual = np.argmax(y_test, axis=1)
print(classification_report(actual, predicts))
