#!/usr/bin/env python3
import time

import cv2
import numpy as np
from scipy import signal

from utils import get_all_eigmin


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


def detect_corners_tomasi(frame, max_corners, min_distance=13, window_size=13):
    """
    Detect corners in a frame using Shi-Tomasi's Corner Detection Algorithm
    Returns an array of corners similar to cv2.goodFeaturesToTrack()
    """
    gkern1d = signal.gaussian(window_size, std=3).reshape(window_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    nrows, ncols = frame.shape
    frame = np.int64(frame)  # cast. Otherwise we'll get overflow
    gx = frame[1:nrows] - frame[0:nrows-1]
    gy = frame[:, 1:ncols] - frame[:, 0:ncols-1]

    # Truncate gx and gy so they are square and have the same shape for element-wise multiplication
    gx = gx[:, 0:ncols-1]
    gy = gy[0:nrows-1, :]
    # show_detected_edges(gx, gy)

    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy

    W_xx = signal.convolve2d(I_xx, gkern2d, mode='same')
    W_xy = signal.convolve2d(I_xy, gkern2d, mode='same')
    W_yy = signal.convolve2d(I_yy, gkern2d, mode='same')

    print('Gonna start getting the eigmins now...')
    eig_start = time.time()
    W = np.stack([W_xx, W_xy, W_xy, W_yy], axis=2)

    eig_mins = get_all_eigmin(W_xx, W_xy, W_yy)
    # eig_mins = np.apply_along_axis(get_eigmin, 2, W)
    print('Finished getting the eigmins in ' + str(time.time() - eig_start) + ' seconds')

    print('Gonna start the mosaicing now...')
    mosaic_start = time.time()
    max_eig_mins = np.zeros(eig_mins.shape)
    for i in range(0, nrows - 1, min_distance):
        for j in range(0, ncols - 1, min_distance):
            i_end = min(i + min_distance, nrows - 1)
            j_end = min(j + min_distance, ncols - 1)

            # last window might not be of size min_distance x min_distance
            window = eig_mins[i:i_end, j:j_end]
            r, c = np.unravel_index(window.argmax(), window.shape)
            max_eig_mins[i + r][j + c] = window[r][c]
    print('Finished mosaicing in ' + str(time.time() - mosaic_start) + ' seconds')

    cutoff_eig_min = np.partition(max_eig_mins.flatten(), -max_corners)[-max_corners]

    row_idxs, col_idxs = np.nonzero(max_eig_mins >= cutoff_eig_min)
    corners = np.vstack((col_idxs, row_idxs)).transpose()
    corners = corners.reshape(corners.size // 2, 1, 2)

    # return value is in the form of (col, row) which corresponds to (x, y)
    return corners


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def builtin_lk(old_frame0, new_frame, corners, lk_params):
    old_gray = to_gray(old_frame)
    new_gray = to_gray(new_frame)

    mask = np.zeros_like(old_frame)
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, corners, None, **lk_params)

    good_old = corners[st == 1]
    good_new = new_corners[st == 1]

    for old, new in zip(good_old, good_new):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(new_frame, (a, b), 5, (0, 255, 0), -1)
    return cv2.add(new_frame, mask)

if __name__ == '__main__':
    # Parameters setup for various processes
    # feature_params = dict(maxCorners=100,
    #                       qualityLevel=0.1,
    #                       minDistance=10)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 1. Read image
    old_frame = cv2.imread('input1.jpg')
    new_frame = cv2.imread('input4.jpg')

    # 2. Detect corners using built-in tracker
    corners1 = detect_corners_tomasi(to_gray(old_frame), 25)
    # compare to this built-in Tomasi corner detector
    # corners1 = cv2.goodFeaturesToTrack(gray1, **feature_params)
    mark_corners(old_frame, corners1)

    # 3. Use built-in optical flow detector (Lucas-Kanade)
    # builtin_lk(old_frame0, new_frame, corners, lk_params)

    # 4. Show the result
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1600, 1200)

    result = old_frame
    cv2.imshow('Result', result)
    while True:
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
