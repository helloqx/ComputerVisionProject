import cv2
import time
import numpy as np
from utils import LEVELS
from pyramid import pyra_down
from single_point_lk import calc_optical_flow_pyr_lk_single


D_THRESHOLD = 5e-2


def calc_optical_flow_pyr_lk(old_frame, new_frame, corners, lk_params, use_original=False):
    if use_original:
        return cv2.calcOpticalFlowPyrLK(old_frame, new_frame, corners, None, **lk_params)

    print('Phase 3: Lucas Kanade Tomasi')
    phase3_start = time.time()

    new_corners = np.zeros_like(corners)
    st = np.ones_like(corners)  # taking all to be successful

    # levels to array index mapping
    # 0 -> original
    # 1 -> original / 2
    # 2... etc etc
    old_frame_levels = [old_frame]
    new_frame_levels = [new_frame]

    for i in range(LEVELS):
        old_frame_levels.append(pyra_down(old_frame_levels[i]))
        new_frame_levels.append(pyra_down(new_frame_levels[i]))

    for idx, corner in enumerate(corners):
        res = calc_optical_flow_pyr_lk_single(old_frame_levels, new_frame_levels, corner)

        new_corners[idx] = res

    # TODO: st should be filtering out corners without movement
    return new_corners, st, None  # err(3rd return value) not used

    print('Phase 3: Lucas Kanade Tomasi, Ended in ' + str(time.time() - phase3_start) + ' seconds')
