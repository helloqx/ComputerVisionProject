from scipy import signal
import numpy as np


WIN_SIZE = 3


def pyra_down(source):

    print(source.shape)
    rows, cols, _ = source.shape
    half_rows = rows >> 1
    half_cols = cols >> 1
    down_source = np.zeros([half_rows, half_cols, 3])
    gkern1d = signal.gaussian(WIN_SIZE, std=1).reshape(WIN_SIZE, 1)
    print(signal.gaussian(WIN_SIZE, std=1))
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= sum(sum(gkern2d))  # normalizing to length 1

    # kernel with normalized gaussian and non-mixing weights on colour channels
    gkern3d = np.reshape(gkern2d, [WIN_SIZE, WIN_SIZE, 1])

    g_source = signal.convolve(source, gkern3d, mode='same')
    for row in range(0, half_rows):
        for col in range(0, half_cols):
            down_source[row, col] = (g_source[row * 2, col * 2])

    return np.uint8(down_source)


def pyra_up(source):
    return source
