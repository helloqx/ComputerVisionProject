import cv2
import time
import numpy as np
from numpy.linalg import LinAlgError
from scipy import signal

D_THRESHOLD = 5e-2

def calc_optical_flow_pyr_lk(old_frame, new_frame, corners, lk_params, use_original=False):
    if use_original:
        return cv2.calcOpticalFlowPyrLK(old_frame, new_frame, corners, None, **lk_params)

    print('Phase 3: Lucas Kanade Tomasi')
    phase3_start = time.time()

    new_corners = np.zeros_like(corners)
    status = np.zeros(len(corners))
    err = 0
    winSize = lk_params.get('winSize')[0]

    gkern1d = signal.gaussian(winSize, std=3).reshape(winSize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= np.sum(gkern2d)  # normalize

    old_frame = np.int32(old_frame)  # cast. Otherwise we'll get overflow
    new_frame = np.int32(new_frame)  # cast. Otherwise we'll get overflow
    Ix = old_frame[:-1, 1:] - old_frame[:-1, :-1]
    Iy = old_frame[1:, :-1] - old_frame[:-1, :-1]

    W_xx = signal.convolve2d(Ix * Ix, gkern2d, mode='same')  # get sum I_xx with Gaussian weight
    W_xy = signal.convolve2d(Ix * Iy, gkern2d, mode='same')  # get sum I_xy with Gaussian weight
    W_yy = signal.convolve2d(Iy * Iy, gkern2d, mode='same')  # get sum I_yy with Gaussian weight

    Z = np.dstack([W_xx, W_xy, W_xy, W_yy])

    I_minus_J = old_frame[:-1,:-1] - new_frame[:-1,:-1]
    I_minus_J_x = I_minus_J * Ix
    I_minus_J_y = I_minus_J * Iy
    W_I_minus_J_x = signal.convolve2d(I_minus_J_x, gkern2d, mode='same')  # get sum(I-J)Ix with Gaussian weight
    W_I_minus_J_y = signal.convolve2d(I_minus_J_y, gkern2d, mode='same')  # get sum(I-J)Iy with Gaussian weight
    
    b = np.dstack([W_I_minus_J_x, W_I_minus_J_y])

    for idx, c in enumerate(corners):
        try:
            c = c.reshape(-1)
            c_int = np.rint(c)
            x = int(c_int[0])
            y = int(c_int[1])
            
            Zc = Z[y, x].reshape(2,2)
            bc = b[y, x].reshape(2,1)
            d = np.linalg.solve(Zc, bc).reshape(-1)

            new_corners[idx] = c + d
            status[idx] = int(d.T.dot(d) > D_THRESHOLD)
        except LinAlgError:
            status[idx] = 0
        except IndexError:
            status[idx] = 0

    print('Phase 3: Lucas Kanade Tomasi, Ended in ' + str(time.time() - phase3_start) + ' seconds')

    return new_corners, status, err
