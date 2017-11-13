import time
import cv2
import numpy as np

from scipy import signal
from utils import get_all_eigmin, to_grayscale


def get_good_features(frame, **kwargs):
    if kwargs.get('use_opencv', False):
        del kwargs['use_opencv']
        return cv2.goodFeaturesToTrack(frame, **kwargs)

    print('Phase 2: Tomasi')
    phase2_start = time.time()

    corners = detect_corners_tomasi(frame, kwargs['maxCorners'], kwargs['minDistance'], kwargs.get('winSize', 13))
    print('Phase 2: Tomasi, Ended in ' + str(time.time() - phase2_start) + ' seconds')

    return corners.astype(np.float32)  # required  by builtin_lk to be of type np.float32


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

    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy

    W_xx = signal.convolve2d(I_xx, gkern2d, mode='same')
    W_xy = signal.convolve2d(I_xy, gkern2d, mode='same')
    W_yy = signal.convolve2d(I_yy, gkern2d, mode='same')

    # DEBUG: show the edges detected
    # show_these = {'Ix': gx, 'Iy': gy, 'W_xx': W_xx, 'W_yy': W_yy}
    # show_images(show_these, normalized=True)

    print('\tGonna start getting the eigmins now...')
    eig_start = time.time()

    eig_mins = get_all_eigmin(W_xx, W_xy, W_yy)
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
    corners = np.vstack((col_idxs, row_idxs)).transpose()
    corners = corners.reshape(corners.size // 2, 1, 2)

    # return value is in the form of (col, row) which corresponds to (x, y)
    return corners
