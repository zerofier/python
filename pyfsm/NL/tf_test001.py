import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(2,),
                          activation='sigmoid'),
    tf.keras.layers.Dense(units=4)
])

batch = np.random.rand(3, 2)

y = model.predict(batch)
print(y.shape)
