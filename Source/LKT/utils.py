import cv2
import numpy as np

EPSILON = 0.15
DEBUG = False
LEVELS = 4
WIN_SIZE = 13  # used by single_point_lk
KEYS = {
    'esc': 27
}
LINE_MAGNITUDE = 1


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_all_eigmin(W_xx, W_xy, W_yy):
    return (W_xx * W_yy - W_xy ** 2) / (W_xx + W_yy + EPSILON)


def mark_corners(frame, corners, size=3, color=(0, 0, 255), with_coords=False):
    """Mark the corners of a frame"""
    frame_copy = np.copy(frame)  # draw the motion in a copied frame, don't mutate the frame itself

    for c in corners:
        x, y = np.rint(c).ravel()
        x, y = int(x), int(y)
        cv2.circle(frame_copy, (x, y), size, color, -1)
        if with_coords:
            # Add the coordinate information if True
            font = cv2.FONT_HERSHEY_SIMPLEX
            coord_text = '({}, {})'.format(x, y)
            cv2.putText(frame_copy, coord_text, (x + 3, y + 3), font, 2, color, thickness=1, lineType=cv2.LINE_4)

    return frame_copy


def mark_motions(frame, old_corners, new_corners, extend_line=False):
    """Mark the motion of the corners: old_corner->good_corner
    :param frame: the frame to draw on
    :param old_corners: old corners
    :param new_corners: corners tracked using motion tracking algorithm
    """
    frame_copy = np.copy(frame)  # draw the motion in a copied frame, don't mutate the frame itself

    for old, new in zip(old_corners, new_corners):
        old_x, old_y = old.ravel()
        new_x, new_y = new.ravel()

        extended_old_x = int(np.rint(old_x - (new_x - old_x) * LINE_MAGNITUDE))
        extended_old_y = int(np.rint(old_y - (new_y - old_y) * LINE_MAGNITUDE))

        new_x = int(np.rint(new_x))
        new_y = int(np.rint(new_y))
        cv2.line(frame_copy, (new_x, new_y), (extended_old_x, extended_old_y), (0, 0, 255), 2)
        cv2.circle(frame_copy, (new_x, new_y), 3, (0, 0, 0), -1)

    return frame_copy


def show_images(image_dict, normalized=False):
    """
    Show images that you pass in image_dict
    :param image_dict: a dict in which the key is the title, and the value is the image to be shown
    :return:
    """

    for k, v in image_dict.items():
        res = v
        if normalized:
            # normalize the color values if flagged True
            res = np.sqrt(v * v)
            res *= 255 / res.max()
            res = np.uint8(res)

        cv2.namedWindow(k, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(k, 1600, 1200)
        cv2.imshow(k, res)
    while True:
        key = cv2.waitKey(0) & 0x00ff
        if key == KEYS['esc']:
            for k in image_dict.keys():
                cv2.destroyWindow(k)
            break


def get_centered_window(frame, x, y, win_size):
    # assumes that pixels out of frame are all 0
    delta = (win_size - 1) // 2

    if x < delta or y < delta:
        raise Exception('X, Y is near edge; Window not full')
    window = frame[x - delta:x + delta, y - delta: y + delta]
    return window
