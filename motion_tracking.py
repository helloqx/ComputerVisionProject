import cv2
import numpy as np


def mark_corners(img, corners):
    """Draw red circles on corners in the image"""
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

if __name__ == '__main__':
    # 1. Read image
    img = cv2.imread('input1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detect corners using built-in tracker
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.1,
                          minDistance=10)
    corners = cv2.goodFeaturesToTrack(gray, **feature_params)
    corners = np.intp(corners)
    mark_corners(img, corners)

    # 3. Use built-in optical flow detector (Lucas-Kanade)

    # 4. Show the result
    cv2.namedWindow('matterhorn', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('matterhorn', 1600, 1200)

    cv2.imshow('matterhorn', img)
    while True:
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
