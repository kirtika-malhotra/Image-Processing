"""import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

try:
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('number_plates'):
        os.makedirs('number_plates')
except OSError:
    print ('Error: Creating directory of data')
currentFrame = 0
shape = "unidentified"
#top
x=0#59
#left
y=0 #71
#right
w=0 #160
#bottom
h=0 #153
flag=1
imCrop=''
cap=cv2.VideoCapture(filename = "E:\\Parking_Hero\\1.MOV")
while(cap.isOpened()):
    ret, frame= cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',frame)
    #fromCenter = False
    #r = cv2.selectROI(frame, fromCenter)
    if(flag==1):
        r = cv2.selectROI(frame)
        imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        x=int(r[1])
        y=int(r[0])
        w=int(r[3])
        h=int(r[2])
        print(x,x+w)
        print(y,y+h)
        cv2.imwrite("roi.png",imCrop)
        gray_roi= cv2.cvtColor(imCrop,cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(src = gray_roi, ksize = (5, 5), sigmaX = 0)
        t_roi, maskLayer_roi= cv2.threshold(src = blur_roi, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
        print(t_roi)
        cv2.destroyWindow("ROI selector")
        
        flag=0
    
   
    
    
    cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),6)
    #cv2.imshow('frame',frame)
   
    currentFrame += 1
   
    if(currentFrame%30==0):
        name = './data/frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name,frame)
        
    
    cv2.namedWindow(winname = "Grayscale Image", flags = cv2.WINDOW_NORMAL)
    cv2.imshow(winname = "Grayscale Image", mat = gray)
    blur = cv2.GaussianBlur(src = gray, ksize = (5, 5), sigmaX = 0)
    (t, maskLayer) = cv2.threshold(src = blur, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    #print(t)
    #cv2.imshow("Threshold",maskLayer)
 

    if(t>t_roi):
        cv2.rectangle(frame,(y,x),(y+h,x+w),(0,255,0),10)
        if(currentFrame%30==0):
            name2 = './number_plates/p' + str(currentFrame) + '.jpg'
            crop= frame[x:x+w,y:y+h]
            cv2.imwrite(name2,crop)
            crop_gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            blur_crop = cv2.GaussianBlur(src = crop_gray, ksize = (5, 5), sigmaX = 0)
            (t_crop, maskLayer_crop) = cv2.threshold(src = blur_crop, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
            contours_crop,hierarchy= cv2.findContours(image=maskLayer_crop,mode= cv2.RETR_EXTERNAL,method= cv2.CHAIN_APPROX_SIMPLE)        
            for cnt in contours_crop:
        #cv2.drawContours(frame, contours = cnt, contourIdx = -1, color = (0, 0, 255), thickness = 5)
                epsilon = 0.01*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                m=cv2.minAreaRect(approx)
                print(m)
                p,q,r = cv2.minAreaRect(approx)
                print(r)
                if(p[1]<q[0]):
                    print("true")
                    cv2.rectangle(crop,(int(p[0]),int(p[1])),(int(q[0]),int(q[1])),(255,0,0),10)
                    crop_im= frame[int(p[0]):int(q[0]),int(p[1]):int(q[1])]
                    cv2.imwrite("cropp.jpg",crop_im)
                    cv2.waitKey(1000)
                print(len(approx))
                #if len(approx) == 4:
			
                 #   (a, b, c, d) = cv2.boundingRect(approx)
                  #  ar = c / float(d)
                   # cv2.drawContours(crop, contours = [cnt], contourIdx = -1, color = (0, 0, 255), thickness = -1)
                    #cv2.waitKey(0)
		    # a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
                    
                """#if(len(approx)==4):
                    #print("rect")
                    #cv2.imwrite("framee",frame)
                    #cv2.drawContours(crop, contours = [cnt], contourIdx = -1, color = (0, 0, 255), thickness = -1)
                    #cv2.waitKey(0)
"""
        

        
        
    else:
        cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
    
    
        #cnt= contours[0]
        #ctr = np.array(cnt).reshape(-1,1,2).astype(np.int8)
            #print(type(cnt))
    #print(type(ctr))
        
        #for (i, c) in enumerate(contours):
            #print("\tSize of contour %d: %d" % (i, len(c)))
    #for c in contours:
        # Returns the location and width,height for every contour
     #       x, y, w, h = cv2.boundingRect(c)
      #      print(x,y,w,h)
        
    cv2.imshow("final",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
"""

import numpy as np
import cv2
import  imutils

# Read the image file
image = cv2.imread('E:\PROJECTS\Python_Codes\number_plates\p33.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None
#we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx #This is our approx Number Plate Contour
        break


# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed
