import numpy as np
from calOpticalFlow import get_centered_window


def test_get_centered_window():
    f = np.reshape(range(16), (4, 4))
    r = get_centered_window(f, 2, 2, 3)
    assert(r,
           [[10, 11],
            [14, 15]])

test_get_centered_window()
