# Description: This file contains the code for the activation function(激活函数) of the neural network.
import matplotlib.pyplot as plt
import numpy as np

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid function 图像
# x=np.arange(-5.0,5.0,0.1)
# y=sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)  # 指定y轴的范围
# plt.show()


# step function
def step_function(x):
    return np.array(x>0,dtype=np.int)  # x>0时返回True，否则返回False

#step function 图像
# x=np.arange(-5.0,5.0,0.1)
# y=step_function(x)
# plt.plot(x,y)
# plt.ylim(-1,2)  # 指定y轴的范围
# plt.show()

# ReLU function
def relu(x):
    return np.maximum(0,x)  # np.maximum()函数会从输入的数值中选择较大的值进行输出

# ReLU function 图像
# x=np.arange(-5.0,5.0,0.1)
# y=relu(x)
# plt.plot(x,y)
# plt.ylim(-1,5)  # 指定y轴的范围
# plt.show()

# identity function
def identity_function(x):
    return x  

# softmax function
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

# softmax function 图像
# a=np.array([0.3,2.9,3.0])
# y=softmax(a)
# print(y)
# plt.plot(a,y)
# plt.show()