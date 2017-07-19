import numpy as np
import cv2
from matplotlib import pyplot as plt


def printImage(image):
    cv2.imshow("window",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Region of Interest in HSV
roi = cv2.imread('car.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)


#Target is the image we search the car in
target = cv2.imread('Park.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)


# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )


# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Convolute with a circular disc
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,kernel,dst)


# Applying a Threshold to the 'Probability' matrix
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))

# Applying a Logical AND operation to maintain only the region of interest
res = cv2.bitwise_and(target,thresh)


cv2.imwrite('res.jpg',res)
printImage(res)



# Kernel created for the following morphological operations
kernel = np.ones((3,3),np.uint8)

# Erosion on resulting image
erosion = cv2.erode(res,kernel,iterations = 3)
printImage(erosion)



# Dilation on Resulting image
dilation = cv2.dilate(erosion,kernel,iterations = 11)
printImage(dilation)

# Painting every white pixel black
dilation[dilation>180] = 1
printImage(dilation)

cv2.imwrite('relatorio3.jpg',dilation)


#Threshold
ret, thresh = cv2.threshold(dilation,100,255,cv2.THRESH_BINARY)
printImage(thresh)


# Second Erosion on threshold image
erosion2 = cv2.erode(thresh,kernel,iterations = 5)
printImage(erosion2)


# Second Dilation on Eroded image
dilation2 = cv2.dilate(erosion2,kernel,iterations = 5)
printImage(dilation2)


























