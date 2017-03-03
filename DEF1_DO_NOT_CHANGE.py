# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:08:55 2017

@author: Mattijs
"""

import numpy as np
import cv2

MIN_MATCH_COUNT = 5;
img1 = cv2.imread('POP_TILES-A3-50x75-BW.jpg',0) # queryImage

img2 = cv2.imread('horizontalmulti.jpg',0) # trainImage
#img2 = cv2.resize(img2,None,fx=2, fy=2, interpolation =  cv2.INTER_AREA)

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=100000)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

######################BRUTEFORCE############################
# create BFMatcher object                                
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
#print des1
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
############################################################

#Convert the array of best matches to an two dimensional array
#So that the best matches can be determined
w, h = 2, len(matches)/2;
Matrix = [[0 for x in range(w)] for y in range(h)] 
                      
for i in range (0, len(matches)/2):
    for j in range (0, 2):
        Matrix[i][j] = matches[i+j]


#tresholding the matches
good = []
for m,n in Matrix:
    if m.distance < 0.97*n.distance:
        good.append(m)

#taking the i.e. the 10 matches with the shortest distance
#better results if it's certain the larger image contains the pattern
#for i in range (1, 25):
#    good.append(matches[i-1])

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    print "Match found!"
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite("IMAGE_NAME.png", img3);
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()