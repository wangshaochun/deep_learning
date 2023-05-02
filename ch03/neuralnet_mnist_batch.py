# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from mnist import load_mnist
from functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


dataset_dir = os.path.dirname(os.path.abspath(__file__))

def init_network():
    with open(dataset_dir+"/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

# 读入数据
x, t = get_data()
# 初始化网络
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0 # 正确识别的数量

for i in range(0, len(x), batch_size):
    # 从x中取出batch_size个数据
    x_batch = x[i:i+batch_size]
    # 预测
    y_batch = predict(network, x_batch)
    # 获取概率最高的元素的索引
    p = np.argmax(y_batch, axis=1)
    # 计算正确识别的数量
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
