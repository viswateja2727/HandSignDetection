import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# while True:
#     success, img = cap.read()

offset=20
imgSize=300
Folder="Data/C"
counter=0
labels=["A","B","C"]
while True:
    success,img=cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)


    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape=imgCrop.shape

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(w*k)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap]=imgResize
            pred, idx = classifier.getPrediction(imgWhite,draw=False)
            print(pred, idx)

        else:
            k = imgSize/w
            hCal = math.ceil(h*k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[ hGap:hCal + hGap,:] = imgResize
            pred, idx = classifier.getPrediction(imgWhite,draw=False)

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
        cv2.rectangle(imgOutput,(x-20,y-70),(x+70,y-20),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[idx],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-20,y-20),(x+w+20,y+h+20),(255,0,255),4)
    cv2.imshow('Video',imgOutput)
    cv2.waitKey(1)


