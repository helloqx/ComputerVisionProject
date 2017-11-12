import cv2
import numpy as np
from numpy.linalg import LinAlgError
from scipy import signal

from corners_tracking import get_good_features
from utils import *

if __name__ == '__main__':
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.1,
                          minDistance=3,
                          use_opencv=False)

    win_size = 13
    gkern1d = signal.gaussian(win_size, std=3).reshape(win_size, 1)
    w_kern = np.ones((win_size, win_size)) #np.outer(gkern1d, gkern1d)
    w_kern = w_kern / np.sum(w_kern)
    
    total_frames, vid_frames = read_video_frames('assets/traffic.mp4')
    corners = get_good_features(to_gray(vid_frames[0]), **feature_params)

    historical_corners = []
    for i in range(0, total_frames-1):
        cur_frame = vid_frames[i]
        nxt_frame = vid_frames[i+1]

        cur_frame_gray = np.int32(to_gray(cur_frame))
        nxt_frame_gray = np.int32(to_gray(nxt_frame))
        
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

        orig_corners = []
        new_corners = []
        ds = []
        for c in corners:
            try:
                c = c.reshape(-1)
                c_int = np.int32(c)
                row = int(c_int[0])
                col = int(c_int[1])
                
                Zc = Z[row, col].reshape(2,2)
                bc = b[row, col].reshape(2,1)
                d = np.linalg.solve(Zc, bc).reshape(-1)

                if abs(d[0]) + abs(d[1]) > 1e-25:
                    orig_corners.append(c)
                    ds.append(d)
                    new_corners.append(c + d)
            except LinAlgError:
                pass
            except IndexError:
                pass

        historical_corners.append(new_corners)
        for cs in historical_corners:
            mark_corners(nxt_frame, cs, size=2, color=(0,0,255))

        if i > 50:
            cv2.imshow('Result %s' % i, nxt_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
