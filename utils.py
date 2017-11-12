import cv2
import numpy as np

EPSILON = 0.15


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_all_eigmin(W_xx, W_xy, W_yy):
    return (W_xx * W_yy - W_xy ** 2) / (W_xx + W_yy + EPSILON)


def mark_corners(img, corners, size=2, color=(0, 0, 255)):
    for c in corners:
        row, col = np.rint(c).ravel()
        cv2.circle(img, (int(row), int(col)), size, color, -1)


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
