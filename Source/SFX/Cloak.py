
# coding: utf-8

# # Invisibility Cloak Special Effect

# In[1]:

import cv2
import numpy as np


# In[2]:

PATH_VIDEO = "ghost.mp4"
PATH_OUTPUT = "output.avi"
PATH_BG = "bg.jpg"


# In[3]:

is_show_scenes = True
is_save_video = False


# In[4]:

def open_video(path=PATH_VIDEO):
    cap = cv2.VideoCapture(PATH_VIDEO)

    # Check if video opened successfully
    if not cap.isOpened(): 
        print("Error opening video stream")
    
    return cap


# In[5]:

def create_video_writer(video):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(PATH_OUTPUT, fourcc, video.get(cv2.CAP_PROP_FPS), 
                         (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# In[6]:

def get_single_frame(video, output_path, frame_num=0):
    '''
    Saves a single frame from video as an image at output_path.
    '''
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cv2.imwrite(output_path, frame)


# In[7]:

bg = cv2.imread(PATH_BG)


# In[19]:

green_brightest_low = np.array([60, 40, 255], np.uint8)
green_brightest_upp = np.array([180, 255, 255], np.uint8)

green_bright_low = np.array([60, 150, 70], np.uint8)
green_bright_upp = np.array([90, 185, 255], np.uint8)

green_dark_low = np.array([60, 185, 70], np.uint8)
green_dark_upp = np.array([90, 235, 255], np.uint8)

green_darkest_low = np.array([60, 235, 70], np.uint8)
green_darkest_upp = np.array([90, 255, 255], np.uint8)


# ## Helper Functions
# _To be replaced with own implementation?_

# In[9]:

def convert_bgr_to_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return hsv


# In[10]:

def threshold_frame(hsv, lower_color, upper_color):
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask


# In[11]:

def invert_frame(frame):
    frame_inv = cv2.bitwise_not(frame)
    return frame_inv


# In[12]:

def mask_frame(frame, mask):
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


# In[13]:

def or_frames(frame1, frame2):
    return cv2.bitwise_or(frame1, frame2)


# ## Edit Video

# In[23]:

cap = open_video()
out = create_video_writer(cap)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        
        hsv = convert_bgr_to_hsv(frame)
        
        # Threshold the HSV image to get only needed colours
        mask_brightest = threshold_frame(hsv, green_brightest_low, green_brightest_upp)
        mask_bright = threshold_frame(hsv, green_bright_low, green_bright_upp)
        mask_dark = threshold_frame(hsv, green_dark_low, green_dark_upp)
        mask_darkest = threshold_frame(hsv, green_darkest_low, green_darkest_upp)
        mask = or_frames(or_frames(mask_brightest, mask_bright), or_frames(mask_darkest, mask_dark))
        
        # Smooth Mask(?)
        kernel = np.ones((5,5),np.uint8)
        # Erode once first to remove some noise
        e1 = cv2.erode(mask, kernel, iterations = 1)
        # DILATE THE HOLES
        mask = cv2.dilate(e1, kernel, iterations = 2)
        
        mask_inv = invert_frame(mask)
        
        # Hide cloth
        res = mask_frame(frame, mask_inv)
        
        # Get "see-through" background
        background = mask_frame(bg, mask)
        
        # Bitwise-OR edited frame and background
        final = or_frames(res, background)

        # Display the resulting frames
        if is_show_scenes:
            cv2.imshow('Original', frame)
            cv2.imshow('Mask', mask)
            #cv2.imshow('BG', background)
            cv2.imshow('Res', res)
            cv2.imshow('Final', final)
 
        if is_save_video:
            out.write(final)

        # Press Q on keyboard to exit
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
 
    # Break the loop when video ends
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# for i in frames:
#     get_single_frame(cap, "./Frames/m" + str(i) + ".jpg", i)

# In[ ]:



