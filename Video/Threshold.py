
# coding: utf-8

# In[1]:

import numpy as np
import cv2


# In[2]:

PATH_IMAGE = "./Frames/f360.jpg"
image = cv2.imread(PATH_IMAGE)


# In[3]:

# dummy function
def nothing(x):
    pass


# In[13]:

# Track bar window
cv2.namedWindow('Threshold Finder')

# Track bars
cv2.createTrackbar('lH','Threshold Finder', 0, 180, nothing)
cv2.createTrackbar('lS', 'Threshold Finder', 0, 255, nothing)
cv2.createTrackbar('lV', 'Threshold Finder', 0, 255, nothing)
cv2.createTrackbar('uH', 'Threshold Finder', 180, 180, nothing)
cv2.createTrackbar('uS', 'Threshold Finder', 255, 255, nothing)
cv2.createTrackbar('uV', 'Threshold Finder', 255, 255, nothing)

while True:

    # converting image color to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Getting values from track bars
    lH = cv2.getTrackbarPos('lH', 'Threshold Finder')
    uH = cv2.getTrackbarPos('uH', 'Threshold Finder')
    lS = cv2.getTrackbarPos('lS', 'Threshold Finder')
    uS = cv2.getTrackbarPos('uS', 'Threshold Finder')
    lV = cv2.getTrackbarPos('lV', 'Threshold Finder')
    uV = cv2.getTrackbarPos('uV', 'Threshold Finder')

    lowerb = np.array([lH, lS, lV], np.uint8)
    upperb = np.array([uH, uS, uV], np.uint8)

    mask = cv2.inRange(hsv, lowerb, upperb)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask= mask)
    cv2.imshow('Threshold Finder', res)

    # Press Q on keyboard to exit
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()


# In[ ]:



