import cv2

image = cv2.imread('../images/images.jpg')
image2 = cv2.imread('../images/detect.png')
image2 = cv2.resize(image2, dsize=(100, 100))

image[50:150, 50:150]=image2
cv2.imshow('img', image)

cv2.waitKey()