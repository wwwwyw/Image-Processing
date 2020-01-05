
import cv2 as cv
import numpy as np


def line_detection(image):
	# 变换为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 进行Canny边缘检测
    edge = cv.Canny(gray, 50, 100, apertureSize=3)
    # 进行霍夫直线运算
    lines = cv.HoughLines(edge, 1, np.pi/180, 200)
    # 对检测到的每一条线段
    for line in lines:
    	# 霍夫变换返回的是 r 和 theta 值
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        # 确定x0 和 y0
        x0 = a * rho
        y0 = b * rho
        # 认为构建（x1,y1）,(x2, y2)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        # 用cv2.line( )函数在image上画直线
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detection", image)


src = cv.imread("image.jpeg")
line_detection(src)
cv.waitKey(0)
cv.destroyAllWindows()