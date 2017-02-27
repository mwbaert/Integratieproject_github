# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:08:55 2017

@author: Mattijs
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 8
print cv2.__version__

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

##FLANN###################################################
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
#############################################################################

##BF############################################ create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
## Match descriptors.
#matches = bf.match(des1,des2)
#
## Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
##############################################


#print matches
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.85*n.distance:
        good.append(m)

print good

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()