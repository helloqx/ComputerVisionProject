
# coding: utf-8

# In[1]:

import cv2
import numpy as np


# In[2]:

PATH_VIDEO = "ghost.mp4"
PATH_OUTPUT = "output.avi"
PATH_BG = "bg.jpg"


# In[3]:

is_show_scenes = False
is_save_video = True


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


# In[8]:

lower_color = np.array([50, 100, 100], np.uint8)
upper_color = np.array([70, 255, 255], np.uint8)


# In[9]:

cap = open_video()
out = create_video_writer(cap)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Threshold the HSV image to get only needed colours
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_inv = cv2.bitwise_not(mask)
        
        # Hide cloth
        res = cv2.bitwise_and(frame, frame, mask=mask_inv)
        
        # Get "see-through" background
        background = cv2.bitwise_and(bg, bg, mask=mask)
        
        # Bitwise-OR edited frame and background
        final = cv2.bitwise_or(res, background)
        
        # Display the resulting frames
        if is_show_scenes:
            cv2.imshow('Frame', frame)
            cv2.imshow('BG', background)
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
cv2.destroyAllWindows()


# In[ ]:



