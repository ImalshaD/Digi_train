import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
dict={}
for i in range(0,10):
    dict[i]=0
for i in range(len(x_train)):
    img = x_train[i]
    lable =y_train[i]
    image = Image.fromarray(np.invert(img), 'L')
    image.save("MNIST\%i\%i.png"%(lable,dict[lable]))
    dict[lable]+=1
    print(i/len(x_train)*100)
for i in range(len(x_test)):
    img = x_test[i]
    lable =y_test[i]
    image = Image.fromarray(np.invert(img), 'L')
    image.save("MNIST\%i\%i.png"%(lable,dict[lable]))
    dict[lable]+=1
    print(i/len(x_train)*100)