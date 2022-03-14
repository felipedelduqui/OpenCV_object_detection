import cv2
import numpy as np
import os

threshold = 25

## Import images

path = 'train'
images = []
classNames = []
orb = cv2.ORB_create(nfeatures=1500)    

myList = os.listdir(path)
print(myList)
print('Total classes detected: ', len(myList))

for cl in myList:
    imgCurrent = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCurrent)
    classNames.append(os.path.splitext(cl)[0]) #removing the ".jpg"

print(classNames)

def findDescriptors(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute (img, None) # Searching key points and descriptors
        desList.append(des) # Adding to our descriptors list
    return desList

def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher() #Brute-force descriptor matchs the descriptions of key points
    matchList = [] #containint the matches for every one of our list
    finalValue = -1 #finalValue will gives the label of the object
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2) #k is the number of values (descriptors?) to compare/match
            good = []
            for m, n in matches:
                if m.distance <0.78*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList)!=0:
        if max(matchList) > threshold:
            finalValue = matchList.index(max(matchList))
    return finalValue


desList = findDescriptors(images)
print('Number of descriptors: ', len(desList))

## Defining camera 
adress = "http://192.168.2.110:8080/video"
cap = cv2.VideoCapture(0)
cap.open(adress) #comment this line, if you want to use an webcam (or cv2.VideoCapture(-1))
cap.set(3, 640)
cap.set(4, 480)

while True:
    sucess, img2 = cap.read()
    imgOriginal = img2.copy() #We will use the original colors for seeing the webcam
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #We will use a gray image to our object detector
    id = findID(img2, desList)

    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 1)

    cv2.imshow('img', imgOriginal)
    cv2.waitKey(1) # 1 millisecond to show it