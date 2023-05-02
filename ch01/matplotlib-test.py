# 用于测试matplotlib是否安装成功
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./images/r.jpeg') # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show() 