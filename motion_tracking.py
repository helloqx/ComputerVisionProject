#!/usr/bin/env python3
import cv2
import numpy as np
from scipy import signal


def mark_corners(img, corners):
    """Draw red circles on corners in the image"""
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


def detect_corners_tomasi(frame, max_corners, min_distance=13, window_size=13):
    """Detect corners in a frame using Shi-Tomasi's Corner Detection Algorithm
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

    # To show the edges detected
    # res = gx + gy
    # res = np.sqrt(res * res)
    # res *= 255 / res.max()
    # res = np.uint8(res)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy

    W_xx = signal.convolve2d(I_xx, gkern2d, mode='same')
    W_xy = signal.convolve2d(I_xy, gkern2d, mode='same')
    W_yy = signal.convolve2d(I_yy, gkern2d, mode='same')

    print('Gonna start getting the eigmins now...')
    nrows, ncols = W_xx.shape
    eig_mins = np.zeros(W_xx.shape)
    for i in range(nrows):
        for j in range(ncols):
            W = np.array([[W_xx[i][j], W_xy[i][j]],
                            [W_xy[i][j], W_yy[i][j]]])
            eig_mins[i][j] = np.amin(np.linalg.eigvals(W))
            # if j == 164: # limiting because it's taking too long
            #     break

    print('Finished getting the eigmins...')
    print('Gonna start the mosaicing now...')
    for i in range(0, nrows, min_distance):
        for j in range(0, ncols, min_distance):
            try:
                window = eig_mins[i:i+min_distance, j:j+min_distance]  # refer to MATLAB->numpy docs for the syntax
            except Exception:  # out of bound exception. TODO: catch a more specific type of Exception
                window = eig_mins[i:nrows, j:ncols]  # last window might not be of size min_distance x min_distance
            r, c = np.unravel_index(window.argmax(), window.shape)
            max_eig_min = window[r][c]
            eig_mins[i:i + min_distance, j:j + min_distance] = np.zeros(window.shape)
            eig_mins[i+r][j+c] = max_eig_min
            # if j > 164:  # limiting because it's taking too long
            #     break
    print('Finished mosaicing...')

    sorted_eig_mins = eig_mins.ravel().copy()
    sorted_eig_mins.sort()
    sorted_eig_mins = sorted_eig_mins[::-1]  # reverse so that it's sorted in descending order
    cutoff_eig_min = sorted_eig_mins[max_corners-1]

    row_idxs, col_idxs = np.nonzero(eig_mins >= cutoff_eig_min)
    corners = np.vstack((col_idxs, row_idxs)).transpose()
    corners = corners.reshape(corners.size // 2, 1, 2)
    return corners  # return value is in the form of (col, row) which corresponds to (x, y)


if __name__ == '__main__':
    # Parameters setup for various processes
    # feature_params = dict(maxCorners=100,
    #                       qualityLevel=0.1,
    #                       minDistance=10)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 1. Read image
    old_frame, new_frame = cv2.imread('input1.jpg'), cv2.imread('input4.jpg')
    gray1, gray2 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # 2. Detect corners using built-in tracker
    corners1 = detect_corners_tomasi(gray1, 25)
    # corners1 = cv2.goodFeaturesToTrack(gray1, **feature_params)  # compare to this built-in Tomasi corner detector
    mark_corners(old_frame, corners1)

    # 3. Use built-in optical flow detector (Lucas-Kanade)
    # mask = np.zeros_like(old_frame)
    # corners2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners1, None, **lk_params)
    #
    # good_new = corners2[st == 1]
    # good_old = corners1[st == 1]
    #
    # for new, old in zip(good_new, good_old):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    #     cv2.circle(new_frame, (a, b), 5, (0, 255, 0), -1)
    # result = cv2.add(new_frame, mask)

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
