import numpy.matlib
import random
import numpy as np
import matplotlib.pyplot as plt
# 直线为y=x+1 x=y-1
from sympy import flatten
# 生成50个符合高斯分布的点作为直线纵坐标
Y=np.matlib.randn(1, 50).A.flatten()
X = Y-1
fig = plt.figure()
# 画图区域分成1行1列。选择第一块区域。
ax1 = fig.add_subplot(1,1, 1)

# 添加一些噪声。
random_x = []
random_y = []
for i in range(50):
    random_x.append(X[i])
    random_y.append(Y[i])
# 添加随机噪声
for i in range(10):
    random_x.append(random.uniform(-2,2))
    random_y.append(random.uniform(-2,2))
RANDOM_X = np.array(random_x) # 散点图的横轴。
RANDOM_Y = np.array(random_y) # 散点图的纵轴。

# 画散点图。
ax1.scatter(RANDOM_X, RANDOM_Y)
# 横轴名称。
ax1.set_xlabel("x")
# 纵轴名称。
ax1.set_ylabel("y")

#计算最小二乘的各个系数
def calcAB(x,y):
    n = len(x)
    sumX, sumY, sumXY, sumXX = 0, 0, 0, 0
    for i in range(0, n):
        sumX += x[i]
        sumY += y[i]
        sumXX += x[i] * x[i]
        sumXY += x[i] * y[i]
    a = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    b = (sumXX * sumY - sumX * sumXY) / (n * sumXX - sumX * sumX)
    return a,b
#得到a b
a,b=calcAB(RANDOM_X,RANDOM_Y)
y1 = a*RANDOM_X+b
plt.plot(RANDOM_X,y1)
plt.title("y = %10.5fx + %10.5f" %(a,b))
plt.show()