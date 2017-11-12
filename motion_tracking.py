#!/usr/bin/env python3

import cv2
import numpy as np

from calOpticalFlow import calc_optical_flow_pyr_lk
from corners_tracking import get_good_features
from utils import *
from pyramid import pyra_down

DISCARD_CRAPPY_CORNERS = True

def main():
    # Parameters setup for various processes
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.1,
                          minDistance=13,
                          use_opencv=False)

    lk_params = dict(winSize=(13, 13),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    total_frames, vid_frames = read_video_frames('assets/traffic.mp4')
    #vid_frames = [cv2.imread('assets/2.jpg'), cv2.imread('assets/1.jpg')]

    frame_index = 0
    tracked_corners = None
    while True:
        # 1. Read image
        old_frame = np.copy(vid_frames[frame_index+1])
        new_frame = np.copy(vid_frames[frame_index])

        # 2. Detect corners using built-in tracker
        if tracked_corners is None:
            tracked_corners = get_good_features(to_gray(old_frame), **feature_params)

        # 3. Use built-in optical flow detector (Lucas-Kanade)
        result, new_corners = lkt(old_frame, new_frame, tracked_corners, lk_params)

        # 4. Show the result
        # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Result', 1600, 1200)
        cv2.destroyAllWindows()
        cv2.imshow('Result %s' % frame_index, result)

        # handling key presses
        k = cv2.waitKey(0) & 0x00ff
        if k == 106:  # j key
            # goes back one frame and resets corners
            frame_index = max(0, frame_index - 1)
            tracked_corners = None
        if k == 107:  # k key
            # stay on current frame and resets corners
            tracked_corners = None
        if k == 108:  # l key
            # move to next frame and reuse corners
            frame_index = min(total_frames - 2, frame_index + 1)
            tracked_corners = new_corners
        if k == 27:  # esc key
            # close program
            cv2.destroyAllWindows()
            break


def lkt(old_frame, new_frame, corners, lk_params):
    old_gray = to_gray(old_frame)
    new_gray = to_gray(new_frame)

    new_corners, st, err = calc_optical_flow_pyr_lk(old_gray, new_gray, corners, lk_params, False)
    # new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, corners, None, **lk_params)

    good_old = corners[st == 1]
    good_new = new_corners[st == 1]

    for old, new in zip(good_old, good_new):
        old_row, old_col = old.ravel()
        new_row, new_col = new.ravel()
        
        delta_row = int(np.rint(old_row - (new_row - old_row) * 10))
        delta_col = int(np.rint(old_col - (new_col - old_col) * 10))

        cv2.line(new_frame, (new_row, new_col), (delta_row, delta_col), (0, 0, 255), 2)
        cv2.circle(new_frame, (new_row, new_col), 2, (0,0,0), -1)

    if DISCARD_CRAPPY_CORNERS:
        new_corners = good_new

    return new_frame, new_corners

if __name__ == '__main__':
    main()
