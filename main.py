import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(1)   #camera number
cap.set(3, 640)
cap.set(4, 480)

totalTablets= 0

myColorFinder = ColorFinder(False)
# Custom Orange Color
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 155, 'hmax': 179, 'smax': 255, 'vmax': 255}
def empty(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings",640,240)
cv2.createTrackbar("Threshold1","Settings",255,255,empty)
cv2.createTrackbar("Threshold2","Settings",120,255,empty)

def preProcessing(img):

    imgPre = cv2.GaussianBlur(img,(5,5),3)
    thresh1= cv2.getTrackbarPos("Threshold1","Settings")
    thresh2= cv2.getTrackbarPos("Threshold2","Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((2, 2), np.uint8)
    imgPre = cv2.dilate(imgPre  , kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

while True:
    success, img = cap.read()
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=20)
    totalTablets =0
    if conFound:
        for count,contour in enumerate(conFound):
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if len(approx)>5:
                area = contour['area']

                if area<4400:
                    totalTablets +=1
                elif 4400<area<5500:
                    totalTablets +=1
                else:
                    totalTablets +=0
                x,y,w,h = contour['bbox']
                imgCrop = img[y:y+h,x:x+w]
                cv2.imshow(str(count), imgCrop)
    imgColor, mask = myColorFinder.update(img, hsvVals)



    print(totalTablets)
    imgStacked = cvzone.stackImages([img, imgPre, imgContours], 2, 1)
    cvzone.putTextRect(imgStacked,f'Tablets.{totalTablets}',(50,50))

    cv2.imshow("Image", imgStacked)
    cv2.imshow("imgColor", imgColor)

    cv2.waitKey(1)    #delay of 1sec
#upar takk camera khul rha tha

