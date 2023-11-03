import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pt

mnist = tf.keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape =(28,28)))
# model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
# # model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.leaky_relu))
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
# model = tf.keras.models.Sequential([
#     # Convolutional and Pooling layers
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     # Flatten and Fully connected layers
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),  # Adding dropout for regularization
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
    
#     # Output layer
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


model = tf.keras.models.Sequential([
    # Convolutional and Pooling layers
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),  # Add BatchNormalization
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten and Fully connected layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32)
model.save('handwritten_digits.model')

loss, accuracy = model.evaluate(x_test,y_test)
print ("loss= ", loss)
print ("Test accuracy= ",accuracy)