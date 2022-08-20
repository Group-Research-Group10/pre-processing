import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#test rgb image
testImg = cv.imread('imgProcessing/TestImage/testFace.jpg',1)

#convert BGR to RGB
originalImg = cv.cvtColor(testImg,cv.COLOR_BGR2RGB)

#croped image to get the face
croppedImg = originalImg[174:530,492:787]

#resized image
ResizeImg = cv.resize(croppedImg,(400,400))

#remove noise
noiseRemoveImg = cv.medianBlur(ResizeImg, 5)

#set brightness to same level
intensityMatrix = np.ones(noiseRemoveImg.shape, dtype='uint8') * 50
setBrightnessImg = cv.subtract(noiseRemoveImg,intensityMatrix)

grayImg = cv.cvtColor(setBrightnessImg, cv.COLOR_RGB2GRAY)


titles = ["Original Image","Cropped Image","Resized Image","Noise Removed Image","Set Brightness","Gray Image"]
images =[originalImg,croppedImg,ResizeImg,noiseRemoveImg,setBrightnessImg,grayImg]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
