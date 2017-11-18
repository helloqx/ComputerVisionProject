#!/usr/bin/env python3
import cv2
import numpy as np

from lucas_kanade import lucas_kanade
from corners_tracking import get_good_features
from utils import *

DISCARD_CRAPPY_CORNERS = False
D_THRESHOLD = 5e-2


def main():
    # Parameters setup for various processes
    feature_params = dict(maxCorners=50,
                          qualityLevel=0.1,
                          minDistance=13,
                          use_opencv=False)

    lk_params = dict(winSize=(13, 13),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 1. Read image
    # old_frame = cv2.imread('assets/input1.jpg')
    # new_frame = cv2.imread('assets/input3.jpg')
    # old_frame = cv2.imread('assets/checkerboard_1.jpg')
    # new_frame = cv2.imread('assets/checkerboard_6.jpg')
    total_frames, vid_frames = read_video_frames('assets/rubbish2.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('rubbish2.mp4', fourcc, 30, (640, 352), isColor=True)

    frame_index = 80
    corners = None
    while frame_index < total_frames-1:
        # 1. Read image
        old_frame = np.copy(vid_frames[frame_index])
        new_frame = np.copy(vid_frames[frame_index + 1])

        blurred_old_frame = cv2.GaussianBlur(old_frame, (7, 7), 5)
        blurred_new_frame = cv2.GaussianBlur(new_frame, (7, 7), 5)

        # 2. Detect corners
        if corners is None:
            corners = get_good_features(to_grayscale(blurred_old_frame), **feature_params)
        if DEBUG:
            corner_detection_result = mark_corners(blurred_old_frame, corners)
            show_images({'Corner detection result': corner_detection_result})

        # 3. Use optical flow detector (Lucas-Kanade)
        tracked_corners, st = lucas_kanade(blurred_old_frame, blurred_new_frame, corners, lk_params)

        good_old = corners
        good_new = tracked_corners
        lk_result = mark_motions(new_frame, good_old, good_new)
        # cv2.imwrite('checkerboard1_and_6.jpg', lk_result)
        # show_images({'Lucas Kanade result': lk_result})

        # 4. Show the result
        # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Result', 1600, 1200)
        # cv2.destroyAllWindows()
        # cv2.imshow('Result %s' % frame_index, lk_result)

        # handling key presses
        # k = cv2.waitKey(0) & 0x00ff
        # if k == 106:  # j key
        #     # goes back one frame and resets corners
        #     frame_index = max(0, frame_index - 1)
        #     corners = None
        # if k == 107:  # k key
        #     # stay on current frame and resets corners
        #     corners = None
        # if k == 108:  # l key
            # move to next frame and reuse corners
        # Define the codec and create VideoWriter object
        out.write(lk_result)
        frame_index += 1
        corners = tracked_corners
        # if k == 27:  # esc key
        #     # close program
        #     cv2.destroyAllWindows()
        #     break

        if DISCARD_CRAPPY_CORNERS:  # for video frames. Can I delete this?
            corners = good_new

    out.release()

if __name__ == '__main__':
    main()
