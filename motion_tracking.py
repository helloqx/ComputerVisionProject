import cv2
import numpy as np


def mark_corners(img, corners):
    """Draw red circles on corners in the image"""
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

if __name__ == '__main__':
    # Parameters setup for various processes
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.1,
                          minDistance=10)
    lk_params = dict(winSize=(15, 15),
                     maxLevel = 2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 1. Read image
    old_frame, new_frame = cv2.imread('input1.jpg'), cv2.imread('input4.jpg')
    gray1, gray2 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # 2. Detect corners using built-in tracker
    corners1 = cv2.goodFeaturesToTrack(gray1, **feature_params)
    # mark_corners(old_frame, corners1)

    # 3. Use built-in optical flow detector (Lucas-Kanade)
    mask = np.zeros_like(old_frame)
    corners2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners1, None, **lk_params)

    good_new = corners2[st == 1]
    good_old = corners1[st == 1]

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(new_frame, (a, b), 5, (0, 255, 0), -1)
    result = cv2.add(new_frame, mask)

    # 4. Show the result
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1600, 1200)

    cv2.imshow('Result', result)
    while True:
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
