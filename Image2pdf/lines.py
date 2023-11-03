import cv2
image = cv2.imread('3.png')
cv2.HoughLinesP(image, rho, theta, threshold, None, minLinLength, maxLineGap)