import numpy as np
from calOpticalFlow import get_centered_window
from pyramid import pyra_down
import cv2


def test_get_centered_window():
    f = np.reshape(range(16), (4, 4))
    r = get_centered_window(f, 2, 2, 3)
    assert(r,
           [[10, 11],
            [14, 15]])

test_get_centered_window()
old_frame = cv2.imread('f1.jpg')
pyra_down_frame = pyra_down(old_frame)

cv2.imshow('original', old_frame)
cv2.imshow('pyra', pyra_down_frame)

while True:
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
        break
