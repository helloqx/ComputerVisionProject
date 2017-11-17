#!/usr/bin/env python3

import time
import cv2
import numpy as np

from pyramid import downsize_frame
from single_point_lk import calc_optical_flow_pyr_lk_single
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
    old_frame = cv2.imread('assets/input1.jpg')
    new_frame = cv2.imread('assets/input2.jpg')
    # old_frame = cv2.imread('assets/checkerboard_1.jpg')
    # new_frame = cv2.imread('assets/checkerboard_2.jpg')

    old_frame = cv2.GaussianBlur(old_frame, (13, 13), 9)
    new_frame = cv2.GaussianBlur(new_frame, (13, 13), 9)

    # 2. Detect corners
    corners = get_good_features(to_grayscale(old_frame), **feature_params)
    if DEBUG:
        corner_detection_result = mark_corners(old_frame, corners)
        show_images({'Corner detection result': corner_detection_result})

    # 3. Use optical flow detector (Lucas-Kanade)
    tracked_corners, st = lucas_kanade(old_frame, new_frame, corners, lk_params)

    # st?
    good_old = corners
    good_new = tracked_corners
    lk_result = mark_motions(new_frame, good_old, good_new)
    show_images({'Lucas Kanade result': lk_result})

    if DISCARD_CRAPPY_CORNERS:  # for video frames. Can I delete this?
        tracked_corners = good_new


def lucas_kanade(old_frame, new_frame, corners, lk_params, use_opencv=False):
    if use_opencv:
        return cv2.calcOpticalFlowPyrLK(old_frame, new_frame, corners, None, **lk_params)

    old_gray = to_grayscale(old_frame)
    new_gray = to_grayscale(new_frame)

    print('Phase 3: Lucas Kanade Tomasi')
    phase3_start = time.time()

    new_corners = np.zeros_like(corners)
    st = np.ones_like(corners)  # taking all to be successful

    # levels to array index mapping
    # 0 -> original
    # 1 -> original / 2
    # 2... etc etc
    old_frame_levels = [old_gray]
    new_frame_levels = [new_gray]

    for i in range(LEVELS):
        old_frame_levels.append(downsize_frame(old_frame_levels[i]))
        new_frame_levels.append(downsize_frame(new_frame_levels[i]))

    for idx, corner in enumerate(corners):
        res = calc_optical_flow_pyr_lk_single(old_frame_levels, new_frame_levels, corner)

        new_corners[idx] = res

    # TODO: st should be filtering out corners without movement
    print('Phase 3: Lucas Kanade Tomasi, Ended in ' + str(time.time() - phase3_start) + ' seconds')

    return new_corners, st


if __name__ == '__main__':
    main()
