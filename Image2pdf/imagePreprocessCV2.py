import cv2
def convert_image_to_grayscale(self):
    return cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
def threshold_image(self):
    return cv2.adaptiveThreshold(self,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
def threshold_image1(self):
    _,img = cv2.threshold(self,130,255,cv2.THRESH_BINARY)
    return img
def invert_image(self):
    return cv2.bitwise_not(self)
def dilate_image(self):
    kernel_size =(2,1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(self, kernel, iterations=1)
def scale_image(self):
    target_size = (28, 28)
    scaled_image = cv2.resize(self, target_size, interpolation=cv2.INTER_AREA)
    return scaled_image
def remove_noice(self):
    kernel_size = (7, 7)  # You can adjust the kernel size based on the noise level
    blurred_image = cv2.GaussianBlur(self, kernel_size, 0)
    return blurred_image
def errode(self):
    kernel_size = (2,2)
    return cv2.erode(self,kernel_size,iterations=2)
# for i in range(1,12001):
image = cv2.imread('WhatsApp Unknown 2023-10-19 at 10.18.26\WhatsApp Image 2023-10-19 at 08.49.47.jpeg')
image1 = convert_image_to_grayscale(image)
image2 = threshold_image(image1)
image2 = threshold_image(image2)

image2 = invert_image(image2)
image3 = dilate_image(image2)
image3 = invert_image(image3)
cv2.imwrite('index.png',image3)