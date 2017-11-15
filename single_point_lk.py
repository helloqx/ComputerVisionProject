import numpy as np
from utils import get_centered_window, DEBUG, WIN_SIZE, LEVELS

LK_REPEATS = 10


def do_lk(old_frame, new_frame, corner, predicted_corner):
    """LK implementation with manual window selection and double summation
    Uses backwards difference in calculation fo Ix, Iy

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
        window_I = get_centered_window(old_frame, corner_x, corner_y, WIN_SIZE)
        # if DEBUG:
        #     print(window_I)
        # window I @ x + 1, y
        window_Ix = get_centered_window(old_frame, corner_x - 1, corner_y, WIN_SIZE)
        # print(window_Ix)
        # window I @ x, y + 1
        window_Iy = get_centered_window(old_frame, corner_x, corner_y - 1, WIN_SIZE)
        # print(window_Iy)

        # using backwards difference, slide 36
        Ix = window_I - window_Ix
        Iy = window_I - window_Iy
        # print(Ix)
        # print(Iy)

        # double summation instead of convolve
        W_xx = np.sum(Ix * Ix)
        W_xy = np.sum(Ix * Iy)
        W_yy = np.sum(Iy * Iy)

        corner_J_y, corner_J_x = map(int, np.rint(predicted_corner[0]))
        window_J = get_centered_window(new_frame, corner_J_x, corner_J_y, WIN_SIZE)

        I_minus_J = window_I - window_J
        I_minus_J_x = I_minus_J * Ix
        I_minus_J_y = I_minus_J * Iy

        W_I_minus_J_x = np.sum(I_minus_J_x)  # get sum (I-J)Ix
        W_I_minus_J_y = np.sum(I_minus_J_y)  # get sum (I-J)Iy

        Z = [[W_xx, W_xy], [W_xy, W_yy]]
        b = [W_I_minus_J_x, W_I_minus_J_y]

        d = np.linalg.solve(Z, b).reshape(-1)
        d = d[::-1]

        return d
    except Exception:
        return 0


def calc_optical_flow_pyr_lk_single(old_frame_levels, new_frame_levels, corner):
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
            new_d = do_lk(
                old_frame_levels[cur_level],
                new_frame_levels[cur_level],
                corner_val_at_levels[cur_level],
                corner_val_at_levels[cur_level] + d
            )
            # update d to as d_updated = d + d_new
            d = new_d + d

    new_corner = corner + d
    return new_corner
