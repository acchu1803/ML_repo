import cv2
import numpy as np

# 1. Capture an image from webcam
cam = cv2.VideoCapture(0)
ret, image = cam.read()
cam.release()

if not ret:
    print("Failed to capture image")
    exit()

cv2.imshow("Captured Image", image)
cv2.imwrite("captured.jpg", image)
cv2.waitKey(0)

resized = cv2.resize(image, (1200, 800))
cv2.imshow("Resized Image", resized)
cv2.imwrite("resized.jpg", resized)
cv2.waitKey(0)

blurred = cv2.blur(image, (50, 50))
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.imwrite("grayscale.jpg", gray)
cv2.waitKey(0)

h, w = image.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1)
rotated = cv2.warpAffine(image, matrix, (w, h))
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)

cv2.destroyAllWindows()