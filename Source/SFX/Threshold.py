
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


# dummy function
def nothing(x):
    pass


# In[3]:


#Read image
PATH_IMAGE = "./Frames/m0.jpg"
image = cv2.imread(PATH_IMAGE)


# In[4]:


def shortcut(x):
    if x == 0:
        cv2.setTrackbarPos('lH', 'Threshold Finder', 0)
        cv2.setTrackbarPos('lS', 'Threshold Finder', 0)
        cv2.setTrackbarPos('lV', 'Threshold Finder', 0)
        cv2.setTrackbarPos('uH', 'Threshold Finder', 180)
        cv2.setTrackbarPos('uS', 'Threshold Finder', 255)
        cv2.setTrackbarPos('uV', 'Threshold Finder', 255)
    elif x == 1:
        cv2.setTrackbarPos('lH', 'Threshold Finder', 0)
        cv2.setTrackbarPos('lS', 'Threshold Finder', 115)
        cv2.setTrackbarPos('lV', 'Threshold Finder', 70)
        cv2.setTrackbarPos('uH', 'Threshold Finder', 80)
        cv2.setTrackbarPos('uS', 'Threshold Finder', 255)
        cv2.setTrackbarPos('uV', 'Threshold Finder', 255)


# In[5]:


def frame_change(x):
    x = x * 30
    global image
    image = cv2.imread("./Frames/m" + str(x) + ".jpg")


# In[14]:


def do_the_thing():
    # Track bar window
    cv2.namedWindow('Threshold Finder')
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    # Track bars
    cv2.createTrackbar('lH','Threshold Finder', 0, 180, nothing)
    cv2.createTrackbar('lS', 'Threshold Finder', 0, 255, nothing)
    cv2.createTrackbar('lV', 'Threshold Finder', 0, 255, nothing)
    cv2.createTrackbar('uH', 'Threshold Finder', 180, 180, nothing)
    cv2.createTrackbar('uS', 'Threshold Finder', 255, 255, nothing)
    cv2.createTrackbar('uV', 'Threshold Finder', 255, 255, nothing)
    cv2.createTrackbar('er', 'Threshold Finder', 0, 3, nothing)
    cv2.createTrackbar('di', 'Threshold Finder', 0, 3, nothing)
    cv2.createTrackbar('quick', 'Threshold Finder', 0, 1, shortcut)
    cv2.createTrackbar('pic', 'Threshold Finder', 0, 31, frame_change)
    

    kernel_o = np.ones((5,5),np.uint8)
    kernel_g = cv2.getGaussianKernel(5, 3)
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
        er = cv2.getTrackbarPos('er', 'Threshold Finder')
        di = cv2.getTrackbarPos('di', 'Threshold Finder')

        lowerb = np.array([lH, lS, lV], np.uint8)
        upperb = np.array([uH, uS, uV], np.uint8)

        mask = cv2.inRange(hsv, lowerb, upperb)

        #erode and dilate
        e1 = cv2.dilate(mask, kernel_g, iterations = di)
        mask = cv2.erode(e1, kernel_g, iterations = er)
        

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hsv, hsv, mask = mask)
        cv2.resizeWindow('Image', 640, 360)
        cv2.imshow('Image', res)

        # Press Q on keyboard to exit
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# In[15]:


do_the_thing()

