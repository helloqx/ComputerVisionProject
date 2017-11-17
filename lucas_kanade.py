import time
import numpy as np
import cv2

from utils import *
from pyramid import downsize_frame

LK_REPEATS = 10


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
        res = get_new_corner(old_frame_levels, new_frame_levels, corner)
        new_corners[idx] = res

    print('Phase 3: Lucas Kanade Tomasi, Ended in ' + str(time.time() - phase3_start) + ' seconds')

    return new_corners, st


def get_new_corner(old_frame_levels, new_frame_levels, corner):
    """Does LKT on the single corner with pyramid

    This algorithm starts off at the deepest depth and at each depth, repeats lk for
    LK_REPEATS number of times to refine the results.

    The located results at each depth is then propogated over to its next higher depth.

    Args:
        old_frame_levels: array size MUST be LEVELS + 1
        new_frame_levels: array size MUST be LEVELS + 1
        corner: the col row of the point of interest
    Return:
        (row, col): calculated position of corner


    """

    # does LEVELS iterations of pyra down
    corner_val_at_levels = [corner]
    for i in range(LEVELS):
        corner_val_at_levels.append(corner_val_at_levels[i] / 2)

    # Initial J is at the same as I
    d = 0
    for cur_level in range(LEVELS, -1, -1):
        d *= 2  # double the d from the previous d
        for i in range(LK_REPEATS):
            new_d = get_d(
                old_frame_levels[cur_level],
                new_frame_levels[cur_level],
                corner_val_at_levels[cur_level],
                corner_val_at_levels[cur_level] + d
            )
            # update d to as d_updated = d + d_new
            d = new_d + d

    new_corner = corner + d
    return new_corner


def get_d(old_frame, new_frame, corner, predicted_corner):
    """LK implementation with manual window selection and double summation
    Uses backwards difference in calculation fo i_x, i_y

    If any of the calculation errors out, then the lk is regarded as a d=0 result.

    This should be almost the same as our old lk with the exception of the variable predicted_corner

    Args:
        old_frame: I frame to be operated on
        new_frame: J frame to be operated on
        corner: the col row of the point of interest, i.e., I(x)
        predicted_corner: the col row of POI in the new frame i.e., J(x)
    Return:
        d: calculated direction of corner from I
    """

    old_frame = np.int32(old_frame)  # cast. Otherwise we'll get overflow
    new_frame = np.int32(new_frame)  # cast. Otherwise we'll get overflow

    corner_y, corner_x = map(int, np.rint(corner[0]))

    # window I @ x, y
    try:
        window_i = get_centered_window(old_frame, corner_x, corner_y, WIN_SIZE)
        window_i_x = get_centered_window(old_frame, corner_x - 1, corner_y, WIN_SIZE)
        window_i_y = get_centered_window(old_frame, corner_x, corner_y - 1, WIN_SIZE)

        # using backwards difference, slide 36
        i_x = window_i - window_i_x
        i_y = window_i - window_i_y

        # double summation instead of convolve
        w_xx = np.sum(i_x * i_x)
        w_xy = np.sum(i_x * i_y)
        w_yy = np.sum(i_y * i_y)

        corner_J_y, corner_J_x = map(int, np.rint(predicted_corner[0]))
        window_j = get_centered_window(new_frame, corner_J_x, corner_J_y, WIN_SIZE)

        i_minus_j = window_i - window_j
        i_minus_j_x = i_minus_j * i_x
        i_minus_j_y = i_minus_j * i_y

        w_i_minus_j_x = np.sum(i_minus_j_x)  # get sum (I-J)i_x
        w_i_minus_j_y = np.sum(i_minus_j_y)  # get sum (I-J)i_y

        z = [[w_xx, w_xy], [w_xy, w_yy]]
        b = [w_i_minus_j_x, w_i_minus_j_y]

        d = np.linalg.solve(z, b).reshape(-1)
        d = d[::-1]

        return d
    except Exception:
        return 0
