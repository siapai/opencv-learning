import cv2

img = cv2.imread("Resources/lambo.png")
imgResize = cv2.resize(img, (300, 200))
imgCropped = img[0:200, 200:500]
cv2.imshow("Original", img)
cv2.imshow("Resized", imgResize)
cv2.imshow("Cropped", imgCropped)
cv2.waitKey(0)
