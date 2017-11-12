#!/usr/bin/env python3

import cv2
import numpy as np

from calOpticalFlow import calc_optical_flow_pyr_lk
from corners_tracking import get_good_features
from utils import to_gray
from pyramid import pyra_down

DISCARD_CRAPPY_CORNERS = True


def main():
    # Parameters setup for various processes
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=10,
                          use_opencv=False)

    lk_params = dict(winSize=(7, 7),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture('assets/traffic.mp4')
    if cap.isOpened() is False:
        print('Error opening video stream or file')

    vid_frames = []
    while cap.isOpened():
        ret, v_frame = cap.read()

        if ret is False:  # reached end of frame
            break
        vid_frames.append(v_frame)

    total_frames = len(vid_frames)
    print('Read %s frames' % total_frames)

    frame_index = 0
    tracked_corners = None
    while True:
        # 1. Read image
        prev_frame = np.copy(vid_frames[frame_index])
        curr_frame = np.copy(vid_frames[frame_index + 1])

        # 2. Detect corners using built-in tracker
        if tracked_corners is None:
            tracked_corners = get_good_features(to_gray(prev_frame), **feature_params)

        # 3. Use built-in optical flow detector (Lucas-Kanade)
        result, new_corners = lkt(prev_frame, curr_frame, tracked_corners, lk_params)

        # mark_corners(prev_frame, tracked_corners)
        # result = prev_frame

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


def mark_corners(img, corners):
    """Draw red circles on corners in the image"""
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


def show_detected_edges(gx, gy):
    # To show the edges detected
    res = gx + gy
    res = np.sqrt(res * res)
    res *= 255 / res.max()
    res = np.uint8(res)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lkt(old_frame, new_frame, corners, lk_params):
    old_gray = to_gray(old_frame)
    new_gray = to_gray(new_frame)

    mask = np.zeros_like(old_frame)

    new_corners, st, err = calc_optical_flow_pyr_lk(old_gray, new_gray, corners, lk_params, False)
    # new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, corners, None, **lk_params)

    good_old = corners[st == 1]
    good_new = new_corners[st == 1]

    for old, new in zip(good_old, good_new):
        new_x, new_y = new.ravel()
        old_x, old_y = old.ravel()
        next_x = new_x + new_x - old_x
        next_y = new_y + new_y - old_y
        # cv2.line(mask, (new_x, new_y), (old_x, old_y), (0, 0, 255), 1)
        cv2.line(mask, (next_x, next_y), (new_x, new_y), (0, 0, 255), 1)
        cv2.circle(new_frame, (new_x, new_y), 1, (0, 255, 0), -1)

    if DISCARD_CRAPPY_CORNERS:
        new_corners = good_new

    return cv2.add(new_frame, mask), new_corners


if __name__ == '__main__':
    main()
