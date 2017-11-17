from scipy import signal
import numpy as np


WIN_SIZE = 3


def downsize_frame(frame):
    is_color = False

    try:
        rows, cols, _ = frame.shape  # color
        half_rows, half_cols = rows // 2, cols // 2
        downsized_frame = np.zeros([half_rows, half_cols, 3])
        is_color = True
    except ValueError:
        rows, cols = frame.shape  # greyscale
        half_rows, half_cols = rows // 2, cols // 2
        downsized_frame = np.zeros([half_rows, half_cols])

    gkern1d = signal.gaussian(WIN_SIZE, std=1).reshape(WIN_SIZE, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= np.sum(gkern2d)  # normalizing the sum to 1

    # kernel with normalized gaussian and non-mixing weights on color channels
    if is_color:
        gkern3d = np.reshape(gkern2d, [WIN_SIZE, WIN_SIZE, 1])
    else:
        gkern3d = gkern2d

    blurred_frame = signal.convolve(frame, gkern3d, mode='same')
    for row in range(0, half_rows):
        for col in range(0, half_cols):
            downsized_frame[row, col] = (blurred_frame[row * 2, col * 2])

    return np.uint8(downsized_frame)
