import tensorflow as tf
import numpy as np

cifar = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# model = tf.keras.applications.MobileNetV2(
#     include_top=True,
#     weights=None,
#     input_shape=(224, 224, 1),
#     classes=10)
# x_train = np.expand_dims(x_train, axis=-1)
# x_train = tf.image.resize(x_train, [224,224]) 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (2,2), input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))   
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=256)