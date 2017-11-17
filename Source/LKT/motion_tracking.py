#!/usr/bin/env python3
import cv2

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
    old_frame = cv2.imread('assets/checkerboard_1.jpg')
    new_frame = cv2.imread('assets/checkerboard_6.jpg')

    old_frame = cv2.GaussianBlur(old_frame, (13, 13), 9)
    new_frame = cv2.GaussianBlur(new_frame, (13, 13), 9)

    # 2. Detect corners
    corners = get_good_features(to_grayscale(old_frame), **feature_params)
    if DEBUG:
        corner_detection_result = mark_corners(old_frame, corners)
        show_images({'Corner detection result': corner_detection_result})

    # 3. Use optical flow detector (Lucas-Kanade)
    tracked_corners, st = lucas_kanade(old_frame, new_frame, corners, lk_params)

    good_old = corners
    good_new = tracked_corners
    lk_result = mark_motions(new_frame, good_old, good_new)
    cv2.imwrite('checkerboard1_and_6.jpg', lk_result)
    show_images({'Lucas Kanade result': lk_result})

    if DISCARD_CRAPPY_CORNERS:  # for video frames. Can I delete this?
        tracked_corners = good_new


if __name__ == '__main__':
    main()
