import time
import cv2
import numpy as np

from scipy import signal
from utils import get_all_eigmin


def get_good_features(frame, **kwargs):

    print('Phase 2: Tomasi')
    phase2_start = time.time()

    corners = detect_corners_tomasi(frame, kwargs['maxCorners'], kwargs['minDistance'], kwargs.get('winSize', 13))
    print('Phase 2: Tomasi, Ended in ' + str(time.time() - phase2_start) + ' seconds')

    return corners.astype(np.float32)


def detect_corners_tomasi(frame, max_corners, min_distance, window_size):
    """
    Detect corners in a frame using Shi-Tomasi's Corner Detection Algorithm
    Returns an array of corners similar to cv2.goodFeaturesToTrack()
    """
    gkern1d = signal.gaussian(window_size, std=3).reshape(window_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    nrows, ncols = frame.shape
    frame = np.int64(frame)  # cast. Otherwise we'll get overflow
    gx = frame[1:nrows] - frame[0:nrows - 1]
    gy = frame[:, 1:ncols] - frame[:, 0:ncols - 1]

    # Truncate gx and gy so they are square and have the same shape for element-wise multiplication
    gx = gx[:, 0:ncols - 1]
    gy = gy[0:nrows - 1, :]
    # show_detected_edges(gx, gy)

    i_xx = gx * gx
    i_xy = gx * gy
    i_yy = gy * gy

    w_xx = signal.convolve2d(i_xx, gkern2d, mode='same')
    w_xy = signal.convolve2d(i_xy, gkern2d, mode='same')
    w_yy = signal.convolve2d(i_yy, gkern2d, mode='same')

    print('\tGonna start getting the eigmins now...')
    eig_start = time.time()

    eig_mins = get_all_eigmin(w_xx, w_xy, w_yy)
    print('\tFinished getting the eigmins in ' + str(time.time() - eig_start) + ' seconds')

    print('\tGonna start the mosaicing now...')
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
    print('\tFinished mosaicing in ' + str(time.time() - mosaic_start) + ' seconds')

    cutoff_eig_min = np.partition(max_eig_mins.flatten(), -max_corners)[-max_corners]

    row_idxs, col_idxs = np.nonzero(max_eig_mins >= cutoff_eig_min)
    row_idxs, col_idxs = row_idxs[:max_corners], col_idxs[:max_corners]
    corners = np.vstack((col_idxs, row_idxs)).transpose()
    corners = corners.reshape(corners.size // 2, 1, 2)

    # return value is in the form of (col, row) which corresponds to (x, y)
    return corners
