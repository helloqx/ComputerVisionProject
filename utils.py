import cv2
import numpy as np

EPSILON = 0.15

KEYS = {
    'esc': 27
}


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_all_eigmin(W_xx, W_xy, W_yy):
    return (W_xx * W_yy - W_xy ** 2) / (W_xx + W_yy + EPSILON)


def mark_corners(img, corners, size=2, color=(0, 0, 255), with_coords=False):
    for c in corners:
        x, y = np.rint(c).ravel()
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, -1)
        if with_coords:
            # Add the coordinate information if True
            font = cv2.FONT_HERSHEY_DUPLEX
            coord_text = '({}, {})'.format(x, y)
            cv2.putText(img, coord_text, (x+3, y+3), font, 0.5, color, thickness=1, lineType=cv2.LINE_4)


def show_detected_edges(gx, gy):
    # To show the edges detected
    res = gx + gy
    res = np.sqrt(res * res)
    res *= 255 / res.max()
    res = np.uint8(res)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_video_frames(file_name):
    """
    NOTE: This function should be deprecated since it's not a good idea to do LK on video
    """
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print("Error opening video stream or file")

    vid_frames = []
    while(cap.isOpened()):
        ret, v_frame = cap.read()

        if not ret:
            # reached end of frame
            break
        vid_frames.append(v_frame)

    total_frames = len(vid_frames)
    print('Read %s frames' % total_frames)
    return total_frames, vid_frames


def show_images(image_dict):
    """
    Show images that you pass in image_dict
    :param image_dict: a dict in which the key is the title, and the value is the image to be shown
    :return:
    """

    for k, v in image_dict.items():
        cv2.namedWindow(k, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(k, 1600, 1200)
        cv2.imshow(k, v)
    while True:
        key = cv2.waitKey(0) & 0x00ff
        if key == KEYS['esc']:
            for k in image_dict.keys():
                cv2.destroyWindow(k)
            break
