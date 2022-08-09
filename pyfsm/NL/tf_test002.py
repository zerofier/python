import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()

y = dataset.target
X = dataset.data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33
)

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=(13, )),
    tf.keras.layers.Dense(units=13, activation='softplus', kernel_regularizer='l1'),
    tf.keras.layers.Dense(units=1)
])
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_train, y_train, epochs=8)

predicts = model.predict(X_test)
result = pd.DataFrame({
    'predict': np.reshape(predicts, (-1, )),
    'actual': y_test
})
limit = np.max(y_test)
result.plot.scatter(x='actual', y='predict', xlim=(0, limit), ylim

=(0, limit))
plt.show()
