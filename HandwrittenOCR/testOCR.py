import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("handwritten_digits.model")
print("Model loaded successfully")
image_number = 1
while os.path.isfile('Images/digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('Images/digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        img = tf.keras.utils.normalize(img)
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1