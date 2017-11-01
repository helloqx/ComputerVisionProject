import cv2


def calc_optical_flow_pyr_lk(old_gray, new_gray, corners, lk_params, use_original=False):
    if use_original:
        return cv2.calcOpticalFlowPyrLK(old_gray, new_gray, corners, None, **lk_params)

    new_corners = 0
    status = 0
    err = 0
    return new_corners, status, err
