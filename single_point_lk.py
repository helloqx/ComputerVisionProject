import numpy as np
from utils import get_centered_window, DEBUG, WIN_SIZE


def do_lk(old_frame, new_frame, corner, predicted_corner):
    """ This should be almost the same as our old lk
    Args
        old_frame
        new_frame
        corner: the col row of the point of interest, i.e., I
        predicted_corner: the col row of POI in the new frame i.e., J
    Returns
        d: calculated direction of corner from I
    """
    old_frame = np.int32(old_frame)  # cast. Otherwise we'll get overflow
    new_frame = np.int32(new_frame)  # cast. Otherwise we'll get overflow

    corner_y, corner_x = map(int, np.rint(corner[0]))
    # window I @ x, y
    try:
        window_I = get_centered_window(old_frame, corner_x, corner_y, WIN_SIZE)
        if DEBUG:
            print(window_I)
        # window I @ x + 1, y
        window_Ix = get_centered_window(old_frame, corner_x + 1, corner_y, WIN_SIZE)
        # print(window_Ix)
        # window I @ x, y + 1
        window_Iy = get_centered_window(old_frame, corner_x, corner_y + 1, WIN_SIZE)
        # print(window_Iy)
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

        W_I_minus_J_x = np.sum(I_minus_J_x)  # get sum(I-J)Ix
        W_I_minus_J_y = np.sum(I_minus_J_y)  # get sum(I-J)Iy

        Z = [[W_xx, W_xy], [W_xy, W_yy]]
        b = [W_I_minus_J_x, W_I_minus_J_y]

        d = np.linalg.solve(Z, b).reshape(-1)
        d = d[::-1]

        # TODO: Figure out why is this negative
        return -d
    except Exception:
        return 0


LEVELS = 4


def calc_optical_flow_pyr_lk_single(old_frame_levels, new_frame_levels, corner):
    """
    Args
        old_frame_levels
        new_frame_levels
        corner: the col row of the point of interest
    Returns
        (row, col): calculated position of corner
    """

    # does 4 iterations of pyra down
    # 0 -> original
    # 1 -> original / 2
    # 2... etc etc
    corner_val_at_levels = [corner]
    for i in range(LEVELS):
        corner_val_at_levels.append(corner_val_at_levels[i] / 2)

    # Initial J is at the same as I
    d = 0
    for cur_level in range(LEVELS, -1, -1):
        new_d = do_lk(
            old_frame_levels[cur_level],
            new_frame_levels[cur_level],
            corner_val_at_levels[cur_level],
            corner_val_at_levels[cur_level] + d
        )
        # update d to be d = 2(d_old + d_found)
        d = new_d + d
        d *= 2

    new_corner = corner + d
    return new_corner
