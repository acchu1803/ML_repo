import cv2
import numpy as np

cam = cv2.VideoCapture(0)
img = cam.read()
cam.release()

if not ret:
    print("Capture failed")
    exit()

cv2.imwrite("captured.jpg", img)

resized = cv2.resize(img, (1200, 800))
blur = cv2.blur(img, (50, 50))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rot = cv2.warpAffine(img, cv2.getRotationMatrix2D)

for title, image in [("Original", img), ("Resized", resized), ("Blurred", blur),
                     ("Grayscale", gray), ("Rotated", rot)]:
    cv2.imshow(title, image)
    cv2.waitKey(0)

cv2.destroyAllWindows()