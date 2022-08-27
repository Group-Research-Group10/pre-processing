from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#faceLandmark dat file
p = "imgProcessing/shape_predictor_68_face_landmarks.dat"

#face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#test rgb image
testImg = cv.imread('imgProcessing/TestImage/testFace.jpg',1)

#convert BGR to RGB
originalImg = cv.cvtColor(testImg,cv.COLOR_BGR2RGB)

#croped image to get the face
croppedImg = originalImg[174:530,492:787]

#resized image
ResizeImg = cv.resize(croppedImg,(500,500))

#remove noise
noiseRemoveImg = cv.medianBlur(ResizeImg, 5)

#set brightness to same level
intensityMatrix = np.ones(noiseRemoveImg.shape, dtype='uint8') * 50
setBrightnessImg = cv.subtract(noiseRemoveImg,intensityMatrix)

#sample Image to get the facial points
sampleForFacialPoints = cv.subtract(noiseRemoveImg,intensityMatrix)

#Gray Image
grayImg = cv.cvtColor(setBrightnessImg, cv.COLOR_RGB2GRAY)


rects = detector(grayImg, 0)

#facial points
for (i, rect) in enumerate(rects):
    shape = predictor(grayImg, rect)
    shape = face_utils.shape_to_np(shape)
    for (x, y) in shape:
        FacialPoints = cv.circle(sampleForFacialPoints, (x, y), 2, (0, 255, 0), -1)


titles = ["Original Image","Cropped Image","Resized Image 500 x 500","Noise Removed Image","Set Brightness","Gray Image","Facial Points"]
images =[originalImg,croppedImg,ResizeImg,noiseRemoveImg,setBrightnessImg,grayImg,FacialPoints]

for i in range(7):
    plt.subplot(2,4,i+1), plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
