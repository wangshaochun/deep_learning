import matplotlib.pyplot as plt
import numpy as np



#初始化network
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])  # 2*3的矩阵
    network['b1']=np.array([0.1,0.2,0.3])  # 1*3的矩阵
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])  # 3*2的矩阵
    network['b2']=np.array([0.1,0.2])  # 1*2的矩阵
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])  # 2*2的矩阵
    network['b3']=np.array([0.1,0.2])  # 1*2的矩阵
    return network

#前向传播
def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    # 第一层
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    # 第二层
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    # 第三层
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    return y

def identity_function(x):
    return x

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)


