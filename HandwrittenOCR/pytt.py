from PIL import Image
import pytesseract
import os
import matplotlib.pyplot as plt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
folder_path="final_out"
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        img = np.invert(np.array([image]))
        print(text)
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()