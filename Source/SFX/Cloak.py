
# coding: utf-8

# # Invisibility Cloak Special Effect

# In[1]:


import cv2
import numpy as np


# In[270]:


PATH_VIDEO = "./Cloak.mov"
PATH_OUTPUT = "output.avi"
PATH_BG = "bg.jpg"
PATH_BG_2 = "bg2.jpg"


# In[328]:


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


def skip_to_frame(video, frame_num):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


# ##### Get background shot
# cap = open_video()
# get_single_frame(cap, "bgasd.jpg", 502)

# In[253]:


bg = cv2.imread(PATH_BG)


# ## Helper Functions

# In[10]:


def convert_bgr_to_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return hsv


# In[11]:


def threshold_frame(hsv, lower_color, upper_color):
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask


# In[12]:


def invert_frame(frame):
    frame_inv = cv2.bitwise_not(frame)
    return frame_inv


# In[13]:


def mask_frame(frame, mask):
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


# In[14]:


def or_frames(frame1, frame2):
    return cv2.bitwise_or(frame1, frame2)


# ## Edit Video

# In[329]:


def process_frame(frameNum, frame):
    '''
        Scene:
        0   - 135    walking in
        135 - 479    unfolding
        480 - 521    cloth over head
        522 - 630    first wear
        631 - 655    pulling over
        656+         completely covered
    '''
    
    if frameNum <= 479:
        return cloth_texture(frame)
    elif frameNum <= 521:
        return protect_hair(frame)
    elif frameNum <= 630:
        return cloth_texture(frame)
    elif frameNum <= 655:
        return protect_hair(frame)
    else:
        return cloth_texture(frame)


# In[259]:


def protect_hair(frame):
        
        green_dk_low = np.array([0, 115, 70], np.uint8)
        green_dk_upp = np.array([80, 255, 150], np.uint8)
        green_nm_low = np.array([0, 115, 150], np.uint8)
        green_nm_upp = np.array([80, 255, 200], np.uint8)
        green_bt_low = np.array([0, 115, 200], np.uint8)
        green_bt_upp = np.array([80, 255, 255], np.uint8)

        head_low = np.array([75, 0, 0], np.uint8)
        head_upp = np.array([100, 255, 255], np.uint8)
        hair_low = np.array([0, 0, 0], np.uint8)
        hair_upp = np.array([180, 255, 77], np.uint8)

        hsv = convert_bgr_to_hsv(frame)
        mask_dk = threshold_frame(hsv, green_dk_low, green_dk_upp)
        mask_nm = threshold_frame(hsv, green_nm_low, green_nm_upp)
        mask_bt = threshold_frame(hsv, green_bt_low, green_bt_upp)
        
        mask_head = threshold_frame(hsv, head_low, head_upp)
        mask_hair = threshold_frame(hsv, hair_low, hair_upp)
        
        mask = or_frames(or_frames(mask_bt, mask_nm), mask_dk)
        
        kernel_ones = np.ones((3,3),np.uint8)
        kernel_gauss = cv2.getGaussianKernel(5, 3)
        
        mask_dil = cv2.erode(mask, kernel_ones, iterations = 1)
        mask_dil = cv2.dilate(mask_dil, kernel_ones, iterations = 5)
        mask_dil = cv2.dilate(mask_dil, kernel_gauss, iterations = 5)
        
        mask_border = cv2.bitwise_and(invert_frame(mask), mask_dil)
        mask_border = cv2.GaussianBlur(mask_border, (5,5), 0)
        
        
        mask_border = cv2.min(invert_frame(mask_hair), mask_border)
        mask_inv = invert_frame(mask_dil)
        
        mask_head = cv2.bitwise_and(mask_dil, mask_head)
        
        mask_inv = cv2.max(mask_inv, mask_head)
        
        og = mask_frame(frame, mask_inv)
        
        # Get "see-through" background
        bg_bt = mask_frame(bg, mask_bt)
        bg_nm = mask_frame(bg, mask_nm)
        bg_dk = mask_frame(bg, mask_dk)
        bg_border = mask_frame(bg, mask_border)
        
        background = mask_frame(bg, mask)
        
        alpha_border = 0.93
        alpha_nm = 0.9
        
        alpha_bt = 0.87
        alpha_d = 0.85
        
        cv2.addWeighted(bg_border, alpha_border, og, 0, 0.0, bg_border);
        cv2.addWeighted(bg_nm, alpha_nm, og, 0, 0.0, bg_nm);
        cv2.addWeighted(bg_bt, alpha_bt, og, 0, 0.0, bg_bt);
        cv2.addWeighted(bg_dk, alpha_d, og, 0, 0.0, bg_dk);
        
        bg_bb = cv2.max(bg_bt, bg_dk)
        bg_bd = cv2.max(bg_border, bg_nm) # BG_BORDER
        bg_all = cv2.max(bg_bb, bg_bd)
        
        final = cv2.max(og, bg_all)
        return final


# In[316]:


def cloth_texture(frame):
        green_dk_low = np.array([0, 115, 70], np.uint8)
        green_dk_upp = np.array([80, 255, 150], np.uint8)
        green_nm_low = np.array([0, 115, 150], np.uint8)
        green_nm_upp = np.array([80, 255, 200], np.uint8)
        green_bt_low = np.array([0, 115, 200], np.uint8)
        green_bt_upp = np.array([80, 255, 255], np.uint8)

        hsv = convert_bgr_to_hsv(frame)
        mask_dk = threshold_frame(hsv, green_dk_low, green_dk_upp)
        mask_nm = threshold_frame(hsv, green_nm_low, green_nm_upp)
        mask_bt = threshold_frame(hsv, green_bt_low, green_bt_upp)
        
        mask = or_frames(or_frames(mask_bt, mask_nm), mask_dk)
        
        kernel_ones = np.ones((3,3),np.uint8)
        kernel_gauss = cv2.getGaussianKernel(5, 3)
        
        mask_dil = cv2.erode(mask, kernel_ones, iterations = 1)
        mask_dil = cv2.dilate(mask_dil, kernel_ones, iterations = 5)
        mask_dil = cv2.dilate(mask_dil, kernel_gauss, iterations = 5)
        mask_border = cv2.bitwise_and(invert_frame(mask), mask_dil)
        mask_border = cv2.GaussianBlur(mask_border, (5,5), 0)
        mask_inv = invert_frame(mask_dil)
        
        og = mask_frame(frame, mask_inv)
        
        # Get "see-through" background
        bg_bt = mask_frame(bg, mask_bt)
        bg_nm = mask_frame(bg, mask_nm)
        bg_dk = mask_frame(bg, mask_dk)
        bg_border = mask_frame(bg, mask_border)
        
        background = mask_frame(bg, mask)
        
        alpha_border = 0.9
        alpha_nm = 0.89
        
        alpha_bt = 0.87
        alpha_d = 0.75
        
        cv2.addWeighted(bg_border, alpha_border, og, 0, 0.0, bg_border);
        cv2.addWeighted(bg_nm, alpha_nm, og, 0, 0.0, bg_nm);
        cv2.addWeighted(bg_bt, alpha_bt, og, 0, 0.0, bg_bt);
        cv2.addWeighted(bg_dk, alpha_d, og, 0, 0.0, bg_dk);
        
        bg_bb = cv2.max(bg_bt, bg_dk)
        bg_bd = cv2.max(bg_border, bg_nm)
        bg_all = cv2.max(bg_bb, bg_bd)
        
        final = cv2.max(og, bg_all)
        return final


# In[321]:


def cloth_texture_2(frame):
        bg = cv2.imread(PATH_BG_2)
        
        green_dk_low = np.array([0, 115, 70], np.uint8)
        green_dk_upp = np.array([80, 255, 150], np.uint8)
        green_nm_low = np.array([0, 115, 150], np.uint8)
        green_nm_upp = np.array([80, 255, 200], np.uint8)
        green_bt_low = np.array([0, 115, 200], np.uint8)
        green_bt_upp = np.array([80, 255, 255], np.uint8)

        hsv = convert_bgr_to_hsv(frame)
        mask_dk = threshold_frame(hsv, green_dk_low, green_dk_upp)
        mask_nm = threshold_frame(hsv, green_nm_low, green_nm_upp)
        mask_bt = threshold_frame(hsv, green_bt_low, green_bt_upp)
        
        mask = or_frames(or_frames(mask_bt, mask_nm), mask_dk)
        
        kernel_ones = np.ones((3,3),np.uint8)
        kernel_gauss = cv2.getGaussianKernel(5, 3)
        
        mask_dil = cv2.erode(mask, kernel_ones, iterations = 1)
        mask_dil = cv2.dilate(mask_dil, kernel_ones, iterations = 5)
        mask_dil = cv2.dilate(mask_dil, kernel_gauss, iterations = 5)
        mask_border = cv2.bitwise_and(invert_frame(mask), mask_dil)
        mask_border = cv2.GaussianBlur(mask_border, (5,5), 0)
        mask_inv = invert_frame(mask_dil)
        
        og = mask_frame(frame, mask_inv)
        
        # Get "see-through" background
        bg_bt = mask_frame(bg, mask_bt)
        bg_nm = mask_frame(bg, mask_nm)
        bg_dk = mask_frame(bg, mask_dk)
        bg_border = mask_frame(bg, mask_border)
        
        background = mask_frame(bg, mask)
        
        alpha_border = 0.9
        alpha_nm = 0.89
        
        alpha_bt = 0.87
        alpha_d = 0.75
        
        cv2.addWeighted(bg_border, alpha_border, og, 0, 0.0, bg_border);
        cv2.addWeighted(bg_nm, alpha_nm, og, 0, 0.0, bg_nm);
        cv2.addWeighted(bg_bt, alpha_bt, og, 0, 0.0, bg_bt);
        cv2.addWeighted(bg_dk, alpha_d, og, 0, 0.0, bg_dk);
        
        bg_bb = cv2.max(bg_bt, bg_dk)
        bg_bd = cv2.max(bg_border, bg_nm)
        bg_all = cv2.max(bg_bb, bg_bd)
        
        final = cv2.max(og, bg_all)
        return final


# In[330]:


cap = open_video()

if is_save_video:
    out = create_video_writer(cap)

skip_to_frame(cap, 0)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        final = process_frame(cap.get(cv2.CAP_PROP_POS_FRAMES), frame)
    
        # Display the resulting frames
        if is_show_scenes:
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Original', 640, 360)
            cv2.imshow('Original', frame)
            
            cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Final', 640, 360)
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
if is_save_video:
    out.release()
cv2.destroyAllWindows()


# In[264]:


cv2.destroyAllWindows()

