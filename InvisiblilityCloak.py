import cv2
import time 
import numpy as np

fourCC = cv2.VideoWriter_fourcc(*'XVID')
outputFile = cv2.VideoWriter("Output.avi", fourCC, 20.0, (1920,1080))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis = 1)
    Hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([36,25,25])
    upper_red = np.array([70,255,255])
    mask1 = cv2.inRange(Hsv, lower_red, upper_red)
    lower_red2 = np.array([110,150,50])
    upper_red2 = np.array([120,255,255])
    mask2 = cv2.inRange(Hsv, lower_red2, upper_red2)
    Mask = mask1 + mask2
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(Mask)
    removebg = cv2.bitwise_and(img, img, mask = mask2)
    replacebg = cv2.bitwise_and(bg, bg, mask = Mask)  
    finalOutput = cv2.addWeighted(removebg, 1, replacebg, 1, 0)
    outputFile.write(finalOutput)
    cv2.imshow("cloack", finalOutput)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()