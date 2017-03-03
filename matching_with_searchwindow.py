# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:56:00 2017

@author: Mattijs
"""
import numpy as np
import cv2

def EvaluateMatch(img1,img2):
    MIN_MATCH_COUNT = 5;
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
        # x, y coordinates of the jeypoints in the source file
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        
        # x, y coordinates of the jeypoints in the destination file
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        #seeking the min and max y-coordinate in the destination file
        #so that the founded pattern/object can be deleted
        #saves the index!!
        max = 0;
        min = img2.shape[0] #height of searching object
        for i in range (0,len(dst_pts)-1):
            print dst_pts[i][0][1]
            if dst_pts[i][0][1] < min:
                min = i
            if dst_pts[i][0][1] > max:
                max = i
                
        print "min :", max

            
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #print pts

        

        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        print "Match found!"
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
   
#######################################################################################################
#MIN_WIDTH_SW = 10; #minimum width of search window
#MIN_HEIGHT_SW = 5; #minimum height of search window
STEP_SIZE_PARAM = 5 #rate that search window increases

print cv2.__version__
singleCarpet = cv2.imread('carpet_1_orig.jpg') # SingleCarpet
frame = cv2.imread('repeat.jpg') # ComposedCarpet

full_width = frame.shape[1] #comparing with MIN_WIDTH_SW this is the maximum
full_height = frame.shape[0] #idem

#concreate increment of searchwindow
step_size_width = full_width/STEP_SIZE_PARAM 
#step_size_height = full_height/STEP_SIZE_PARAM

current_width = step_size_width
#current_height = step_size_height



#full image loaded
full_flag = 0; 
teller = 0;
while(full_flag == 0):
    teller = teller + 1
    #searchwindow = frame[0:current_height, 0:current_width]
    searchwindow = frame[0:full_height, 0:current_width]
    #current_height = current_height + step_size_height
    current_width = current_width + step_size_width
    EvaluateMatch(singleCarpet,searchwindow)
#    cv2.imshow('image',searchwindow)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    if teller == STEP_SIZE_PARAM:
        full_flag = 1
    



