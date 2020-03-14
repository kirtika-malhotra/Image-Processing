# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2
from PIL import Image

cap = cv2.VideoCapture('E:\\IGDTUW\\Parking_Hero\\DATA\\images,gifs,videos\\new video.mp4') 
fgbg = cv2.createBackgroundSubtractorMOG2(history=1,varThreshold=1000,detectShadows=False)

flag=1
x=''
y=''
w=''
h=''
coordinates=[]
inn_coords=[]
imCrop_roi=''
imCrop_frame=[]
in_char='y'
n_white_pix=[]
while(cap.isOpened()): 
    ret, frame = cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if(flag==1):
        while(in_char=='y'):
            showCrosshair = False
            fromCenter = False
            cv2.namedWindow("roi",2)
            r = cv2.selectROI("roi", frame, fromCenter, showCrosshair)
            imCrop_roi= frame[int(r[1]) : int(r[1])+int(r[3]) , int(r[0]) : int(r[0])+ int(r[2])]
            x=int(r[1])
            y=int(r[0])
            w=int(r[3])
            h=int(r[2])
            inn_coords.append(x)
            inn_coords.append(y)
            inn_coords.append(w)
            inn_coords.append(h)

            coordinates.append(inn_coords)
            inn_coords=[]
            cv2.rectangle(frame,(y,x),(y+h,w+x),(0,255,0),5)
            cv2.destroyWindow("roi")

            in_char=input("select more?")
            
        print(coordinates)
        
        flag=0

    for coords in coordinates:
        x=coords[0]
        y=coords[1]
        w=coords[2]
        h=coords[3]
        cv2.rectangle(frame,(y,x),(y+h,w+x),(0,255,0),5)
        imCrop_frame=gray[x:x+w,y:y+h]
        cv2.imshow("crop",imCrop_frame)
        fgmask = fgbg.apply(imCrop_frame)
        cv2.imshow("mask",fgmask)
        #cv2.waitKey(0)
        n_white_pix=(np.sum(fgmask == 255))
        print(n_white_pix)
        if(n_white_pix > 400):
            print("red")
            cv2.rectangle(frame,(y,x),(y+h,w+x),(0,0,255),5)
            #n_white_pix=[]
        #print("car")
    #print(n_white_pix)
        
    #cv2.rectangle(frame,(y,x),(y+h,w+x),(0,0,255,10))
    #fgmask= imCrop_roi - frame
    #cv2.imshow('fg', foregrnd)
    cv2.namedWindow("frame",2)
    cv2.imshow('frame',frame)    
    #cv2.imshow('gray', imCrop_frame) 
    #cv2.imshow("mask",fgmask)
      
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
      
  
cap.release()
cv2.destroyAllWindows()
