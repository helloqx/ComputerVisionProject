import cv2
import numpy as np

# 1. Read image
img = cv2.imread('flower_pot_gray.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Detect corners using built-in tracker
corners = cv2.goodFeaturesToTrack(gray,200,0.01,10)
corners = np.intp(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 7, (0, 0, 255), -1)

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
