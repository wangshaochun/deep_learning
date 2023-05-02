import os, sys
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 将NumPy数组转换为图像
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img=x_train[0]
label=t_train[0]
print(label) # 5

print(img.shape) # (784,)
img=img.reshape(28,28) # 把图像的形状变成原来的尺寸(数组)
print(img) # (28, 28)
img_show(img)