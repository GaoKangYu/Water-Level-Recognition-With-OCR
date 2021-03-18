import cv2 as cv
import numpy as np

lane = cv.imread("TestImg/9.jpg")
raw = lane
cv.namedWindow("raw", 0)
cv.namedWindow("lane", 0)
cv.namedWindow("line_img", 0)
#cv.resizeWindow("raw", 640, 480)
cv.imshow("raw", raw)
ori = lane
# 高斯模糊，Canny边缘检测需要的
lane = cv.GaussianBlur(lane, (5, 5), 0)
# 进行边缘检测，减少图像空间中需要检测的点数量
lane = cv.Canny(lane, 50, 150)
cv.imshow("lane", lane)
rho = 1  # 距离分辨率
theta = np.pi / 180  # 角度分辨率
threshold = 30  # 霍夫空间中多少个曲线相交才算作正式交点
min_line_len = 30  # 最少多少个像素点才构成一条直线
max_line_gap = 30  # 线段之间的最大间隔像素
lines = cv.HoughLinesP(lane, rho, theta, threshold, maxLineGap=max_line_gap)
line_img = np.zeros_like(lane)
i = 1
for line in lines:
    for x1, y1, x2, y2 in line:
        print(x1,x2,y1,y2)
        cv.line(ori, (x1, y1), (x2, y2), 255, 1)
        cv.putText(ori, str(i), (x2, y2), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1)
        i = i + 1
print('检测到了', int(lines.shape[0]/2), '条直线')
cv.putText(ori, 'Line Number:' + str(int(lines.shape[0]/2)), (80, 20), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1)
cv.imshow("line_img", ori)
cv.waitKey()