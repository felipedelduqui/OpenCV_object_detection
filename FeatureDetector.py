import cv2
import numpy as np

#Trains
sophie1_train = cv2.imread('train/sophie1.jpg', 0) #0 means gray scale, to speed up the process
sophie2_train = cv2.imread('train/sophie2.jpg', 0)
ffx_train = cv2.imread('train/ffx.jpg', 0)
ffxii_train = cv2.imread('train/ffxii.jpg', 0)
tbrise_train = cv2.imread('train/tbrise.jpg', 0)
tbshadow_train = cv2.imread('train/tbshadow.jpg', 0)
firis_train = cv2.imread('train/firis.jpg', 0)
p5s_train = cv2.imread('train/p5s.jpg', 0)

#Tests
sophie1_test = cv2.imread('test/sophie1.jpg', 0)
sophie2_test = cv2.imread('test/sophie2.jpg', 0)
ffx_test = cv2.imread('test/ffx.jpg', 0)
ffxii_test = cv2.imread('test/ffxii.jpg', 0)
tbrise_test = cv2.imread('test/tbrise.jpg', 0)
tbshadow_test = cv2.imread('test/tbshadow.jpg', 0)
firis_test = cv2.imread('test/firis.jpg', 0)
p5s_test = cv2.imread('test/p5s.jpg', 0)

#Tests resize

sophie1_test = cv2.resize(sophie1_test, (756,1008))
sophie2_test = cv2.resize(sophie2_test, (756,1008))
ffx_test = cv2.resize(ffx_test, (756,1008))
ffxii_test = cv2.resize(ffxii_test, (756,1008))
tbrise_test = cv2.resize(tbrise_test, (756,1008))
tbshadow_test = cv2.resize(tbshadow_test, (756,1008))   
firis_test = cv2.resize(firis_test, (756,1008))
p5s_test = cv2.resize(p5s_test, (756,1008))

orb = cv2.ORB_create(nfeatures=1500) # Model ORB, a fast and free model, n_features by default is 500

kp_s1test, des_s1test= orb.detectAndCompute(sophie1_test, None) # key point test/ descriptor -> features
kp_s1train, des_s1train= orb.detectAndCompute(sophie1_train, None) # key point train/ descriptor -> features


#Key points draws
img_kp_s1test = cv2.drawKeypoints(sophie1_test, kp_s1test, None)
img_kp_s1train = cv2.drawKeypoints(sophie1_train, kp_s1train, None)

bf = cv2.BFMatcher() #Brute-force descriptor matchs the descriptions of key points
matches = bf.knnMatch(des_s1test, des_s1train, k=2) #k is the number of values to compare


good = []
for m, n in matches:
    if m.distance <0.78*n.distance:
        good.append([m])

print(len(good))

img_matches = cv2.drawMatchesKnn(sophie1_test, kp_s1test, sophie1_train, kp_s1train, good, None, flags=2)
img_matches = cv2.resize(img_matches, (1205,710))

#None means no out image (?)
#flags means "how you want to show"

#Show images

#cv2.imshow('Key Point from Atelier Sophie test', img_kp_s1test)
#cv2.imshow('Key Point from Atelier Sophie train', img_kp_s1train)
cv2.imshow('Matching good descriptions', img_matches)
cv2.imshow('Atelier Sophie', sophie1_test)
cv2.imshow('Atelier Sophie 2', sophie2_test)
cv2.waitKey(0)