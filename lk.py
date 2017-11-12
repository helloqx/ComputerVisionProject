import cv2
import numpy as np
from scipy import signal

from corners_tracking import get_good_features
from utils import *

def mark_corners(img, corners, size=2, color=(0, 0, 255)):
    for c in corners:
        y, x = np.int32(c).ravel()
        cv2.circle(img, (y, x), size, color, -1)

if __name__ == '__main__':
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.1,
                          minDistance=10,
                          use_opencv=False)

    vid_frames = read_video_frames('assets/traffic.mp4')

    win_size = 13
    gkern1d = signal.gaussian(win_size, std=3).reshape(win_size, 1)
    w_kern = np.outer(gkern1d, gkern1d)
    
    corners = get_good_features(to_gray(vid_frames[0]), **feature_params)

    i = 0
    historical_corners = []
    while True:
        cur_frame = np.copy(vid_frames[i])
        nxt_frame = np.copy(vid_frames[i+1])

        cur_frame_gray = to_gray(cur_frame)
        nxt_frame_gray = to_gray(nxt_frame)

        I_x = (cur_frame_gray[:-1, :-1] - cur_frame_gray[:-1, 1:])
        I_y = (cur_frame_gray[:-1, :-1] - cur_frame_gray[1:, :-1])
        
        I2_x = signal.convolve2d(I_x**2, w_kern, mode='same')
        I_xI_y = signal.convolve2d(I_x * I_y, w_kern, mode='same')
        I2_y = signal.convolve2d(I_y**2, w_kern, mode='same')

        Z = np.dstack([I2_x, I_xI_y, I_xI_y, I2_y])

        IminJ = cur_frame_gray[:-1,:-1] - nxt_frame_gray[:-1,:-1]
        IminJ_Ix = signal.convolve2d(IminJ * I_x, w_kern, mode='same')
        IminJ_Iy = signal.convolve2d(IminJ * I_y, w_kern, mode='same')

        b = np.dstack([IminJ_Ix, IminJ_Iy])

        new_corners = []
        for c in corners:
            try:
                c = c.reshape(-1)
                c_int = np.int32(c)
                x = int(c_int[0])
                y = int(c_int[1])
                
                Zc = Z[y,x].reshape(2,2)
                bc = b[y,x].reshape(2,1)
                d = np.linalg.solve(Zc, bc).reshape(-1)[::-1]

                if abs(d[0]) + abs(d[1]) > 1e-25:
                    new_corners.append(c + d)
            except IndexError:
                pass

        historical_corners.append(new_corners)

        if i < 15:
            i = i+1
            corners = new_corners
        else:
            for idx, cs in enumerate(historical_corners):
                mark_corners(nxt_frame, cs, size=2, color=(0,0,255,idx/len(cs)))
            
            cv2.destroyAllWindows()
            cv2.imshow('Result %s' % i, nxt_frame)

            # handling key presses
            k = cv2.waitKey(0) & 0x00ff
            if k == 107:  # k key
                corners = get_good_features(to_gray(vid_frames[i]), **feature_params)
            if k == 108:  # l key
                i = i + 1
                corners = new_corners
            if k == 27:  # esc key
                cv2.destroyAllWindows()
                break

            