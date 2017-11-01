import cv2
import time
import numpy as np
from scipy import signal


def get_centered_window(frame, x, y, win_size):
    # assumes that pixels out of frame are all 0

    delta = (win_size - 1) << 1
    print(delta, x - delta, x + delta)
    window = frame[x - delta:x + delta][:, y - delta: y + delta]

    return window


def calc_optical_flow_pyr_lk(old_frame, new_frame, corners, lk_params, use_original=False):
    if use_original:
        return cv2.calcOpticalFlowPyrLK(old_frame, new_frame, corners, None, **lk_params)

    print('Phase 3: Lucas Kanade Tomasi')
    phase3_start = time.time()

    new_corners = np.zeros_like(corners)
    status = np.zeros(len(corners))
    err = 0
    winSize = lk_params.get('winSize')[0]
    frame_rows, frame_cols = old_frame.shape
    print(frame_rows, frame_cols)
    gkern1d = signal.gaussian(winSize, std=3).reshape(winSize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    gx = old_frame[1:frame_rows] - old_frame[0:frame_rows - 1]
    gy = old_frame[:, 1:frame_cols] - old_frame[:, 0:frame_cols - 1]

    # Truncate gx and gy so they are square and have the same shape for element-wise multiplication
    Ix = gx[:, 0:frame_cols - 1]
    Iy = gy[0:frame_rows - 1, :]

    I_xx = Ix * Ix
    I_xy = Ix * Iy
    I_yy = Iy * Iy

    W_xx = signal.convolve2d(I_xx, gkern2d, mode='same')  # get sum I_xx with Gaussian weight
    W_xy = signal.convolve2d(I_xy, gkern2d, mode='same')  # get sum I_xy with Gaussian weight
    W_yy = signal.convolve2d(I_yy, gkern2d, mode='same')  # get sum I_yy with Gaussian weight

    I_minus_J = old_frame - new_frame
    I_minus_J = I_minus_J[0:frame_rows - 1, 0: frame_cols - 1]
    I_minus_J_x = I_minus_J * Ix
    I_minus_J_y = I_minus_J * Iy
    W_I_minus_J_x = signal.convolve2d(I_minus_J_x, gkern2d, mode='same')  # get sum(I-J)Ix with Gaussian weight
    W_I_minus_J_y = signal.convolve2d(I_minus_J_y, gkern2d, mode='same')  # get sum(I-J)Iy with Gaussian weight
    """
    I = old frame, J = new frame
    I(x) = old frame at position x
    J(x) = new frame at position x

    for each corner
        get the window centered around corner, win=winSize
        get sum I_xx with Gaussian weight
        get sum I_xy with Gaussian weight
        get sum I_yy with Gaussian weight
        get sum(I-J)Ix with Gaussian weight
        get sum(I-J)Iy with Gaussian weight
    """
    for idx, corner in enumerate(corners):
        corner = corner
        x, y = map(int, corner[0])
        # [Z b]
        Z = [
            [W_xx[x, y], W_xy[x, y]],
            [W_xy[x, y], W_yy[x, y]]
        ]
        b = [[W_I_minus_J_x[x, y]], [W_I_minus_J_y[x, y]]]
        # using linagl.solve
        d = np.linalg.solve(Z, b)
        # SOLVE for [Z b]
        # Zb = np.hstack((Z, b))
        # U, S, V = np.linalg.svd(Zb)
        # d = V * ((np.transpose(U) * b) / np.diag(S))
        # inner = np.transpose(U) * b
        # rhs = inner / np.diag(S)
        # d = V * rhs
        # print(corner, d)
        new_corners[idx] = corner + np.transpose(d)
        status[idx] = 1

    print('Phase 3: Lucas Kanade Tomasi, Ended in ' + str(time.time() - phase3_start) + ' seconds')

    return new_corners, status, err
