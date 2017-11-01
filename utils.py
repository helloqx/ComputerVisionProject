import cv2
EPSILON = 0.15


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_all_eigmin(W_xx, W_xy, W_yy):
    return (W_xx * W_yy - W_xy ** 2) / (W_xx + W_yy + EPSILON)
