import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("handwritten_digits.model")
print("Model loaded successfully")
folder_path="final_out"
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)[:,:,0]
        img = np.invert(np.array([img]))
        img = tf.keras.utils.normalize(img)
        prediction = model.predict(img)
        print(prediction)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()