#!/usr/bin/env python3

import cv2
import numpy as np

from cal_optical_flow import calc_optical_flow_pyr_lk
from corners_tracking import get_good_features
from utils import to_grayscale, DEBUG, show_images

DISCARD_CRAPPY_CORNERS = False
LINE_MAGNITUDE = 15


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
    old_frame = cv2.imread('assets/input1.jpg')
    new_frame = cv2.imread('assets/input2.jpg')
    # old_frame = cv2.imread('assets/checkerboard_1.jpg')
    # new_frame = cv2.imread('assets/checkerboard_2.jpg')

    old_frame = cv2.GaussianBlur(old_frame, (13, 13), 9)
    new_frame = cv2.GaussianBlur(new_frame, (13, 13), 9)

    # 2. Detect corners
    tracked_corners = get_good_features(to_grayscale(old_frame), **feature_params)

    # 3. Use optical flow detector (Lucas-Kanade)
    result, new_corners = lkt(old_frame, new_frame, tracked_corners, lk_params)
    if DEBUG:
        show_images({'LK': result})


def lkt(old_frame, new_frame, corners, lk_params):
    old_gray = to_grayscale(old_frame)
    new_gray = to_grayscale(new_frame)

    new_corners, st, err = calc_optical_flow_pyr_lk(old_gray, new_gray, corners, lk_params, use_original=False)

    good_old = corners
    good_new = new_corners

    for old, new in zip(good_old, good_new):
        old_x, old_y = old.ravel()
        new_x, new_y = new.ravel()

        extended_old_x = int(np.rint(old_x - (new_x - old_x) * LINE_MAGNITUDE))
        extended_old_y = int(np.rint(old_y - (new_y - old_y) * LINE_MAGNITUDE))

        new_x = int(np.rint(new_x))
        new_y = int(np.rint(new_y))
        cv2.line(new_frame, (old_x, old_y), (extended_old_x, extended_old_y), (0, 0, 255), 2)
        cv2.circle(new_frame, (old_x, old_y), 3, (0, 0, 0), -1)

    if DISCARD_CRAPPY_CORNERS:
        new_corners = good_new

    return new_frame, new_corners


if __name__ == '__main__':
    main()
