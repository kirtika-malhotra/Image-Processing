# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2
from PIL import Image
"""
im1= cv2.imread("roi_new.png")
im2= cv2.imread("car.png")
im3= im1-im2
fgbg = cv2.createBackgroundSubtractorMOG2() 
fgmask = fgbg.apply(im2)

  
while(1):
    cv2.imshow("im",im3)
    cv2.imshow('frame', fgmask) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
      
  

cv2.destroyAllWindows()
    
"""
cap = cv2.VideoCapture('E:\\IGDTUW\\Parking_Hero\\DATA\\images,gifs,videos\\parking.MOV') 
fgbg = cv2.createBackgroundSubtractorMOG2(history=1,varThreshold=1000,detectShadows=False)
#print(fgbg)
#im1= cv2.imread("roi_new.png")
#print(im1.shape)



flag=1
i=''
x=''
y=''
w=''
h=''
imCrop_roi=''
in_char='Y'
while(cap.isOpened()): 
    ret, frame = cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    coordinates=[]
    inn_coords=[]
    if(flag==1):
        while(in_char=='y' or in_char=='Y'):
            showCrosshair = False
            fromCenter = False
            cv2.namedWindow("roi",2)
            r = cv2.selectROI("roi", frame, fromCenter, showCrosshair)
             
            #r= cv2.selectROI("frame",frame,False,False)
            
            imCrop_roi= frame[int(r[1]) : int(r[1])+int(r[3]) , int(r[0]) : int(r[0])+ int(r[2])]
            x=int(r[1])
            y=int(r[0])
            w=int(r[3])
            h=int(r[2])
            inn_coords.append(x)
            inn_coords.append(y)
            inn_coords.append(w)
            inn_coords.append(h)
            
            print(inn_coords)
            coordinates.append(inn_coords)
            print(coordinates)
            #cv2.imwrite("roi.png",imCrop_roi)
            
            in_char=input("select more??")
           
        cv2.destroyWindow("roi")
        flag=0

    for i in coordinates:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]
        #cv2.rectangle(frame,(i[1],i[0]),(i[1]+i[3],i[2]+i[0]),(0,255,0),10)
        #imCrop_frame= gray[i[0]:i[0]+i[2],i[1]:i[1]+i[3]]
    imCrop_frame= gray[x:x+w,y:y+h]
    
    cv2.rectangle(frame,(y,x),(y+h,w+x),(0,255,0),5)
    cv2.namedWindow('frame',2)
    cv2.imshow('frame',frame)
    #cv2.rectangle(frame,(964,653),(964+150,155+653),(0,255,0),5)
    fgmask = fgbg.apply(imCrop_frame)
    
    n_white_pix = np.sum(fgmask == 255)
    
    if(n_white_pix > 400):
        print("car")
        print(n_white_pix)
        cv2.rectangle(frame,(y,x),(y+h,w+x),(0,0,255),5)
    #cv2.rectangle(frame,(y,x),(y+h,w+x),(0,0,255,10))
    #fgmask= imCrop_roi - frame
    #cv2.imshow('fg', foregrnd)
    cv2.namedWindow('frame',2)
    cv2.imshow('frame',frame)
    cv2.namedWindow('gray',2)
    cv2.imshow('gray', imCrop_frame)
    cv2.namedWindow('mask',2)
    cv2.imshow("mask",fgmask)
      
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
      
  
cap.release()
cv2.destroyAllWindows()
