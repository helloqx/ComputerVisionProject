#!/usr/bin/env python3

import cv2
import numpy as np

from calOpticalFlow import calc_optical_flow_pyr_lk
from corners_tracking import get_good_features
from utils import to_gray


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

    good_old = corners[st == 1]
    good_new = new_corners[st == 1]

    for old, new in zip(good_old, good_new):
        newX, newY = new.ravel()
        oldX, oldY = old.ravel()
        cv2.line(mask, (newX, newY), (oldX, oldY), (0, 255, 0), 1)
        cv2.circle(new_frame, (newX, newY), 1, (0, 255, 0), -1)
    return cv2.add(new_frame, mask)

if __name__ == '__main__':
    # Parameters setup for various processes
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=10,
                          use_opencv=False)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 1. Read image
    old_frame = cv2.imread('f1.jpg')
    new_frame = cv2.imread('f2.jpg')
    # old_frame = cv2.imread('input1.jpg')
    # new_frame = cv2.imread('input2.jpg')

    # 2. Detect corners using built-in tracker
    corners1 = get_good_features(to_gray(old_frame), **feature_params)
    # mark_corners(old_frame, corners1)
    # result = old_frame

    # 3. Use built-in optical flow detector (Lucas-Kanade)
    result = lkt(old_frame, new_frame, corners1, lk_params)

    # 4. Show the result
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1600, 1200)

    cv2.imshow('Result', result)
    while True:
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
