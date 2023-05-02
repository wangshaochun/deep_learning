# Desciption: 感知机实现与门、或门、与非门、异或门
import numpy as np

#AND 与门
def AND(x1,x2):
    x=np.array([x1,x2])
    y=np.array([0.5,0.5])
    b= -0.7
    w=np.sum(x*y)+b
    if w<=0:
        return 0
    else:
        return 1

#OR 或门
def OR(x1,x2):
    x=np.array([x1,x2])
    y=np.array([0.5,0.5])
    b= -0.2
    w=np.sum(x*y)+b
    if w<=0:
        return 0
    else:
        return 1

#NAND 与非门
def NAND(x1,x2):
    x=np.array([x1,x2])
    y=np.array([-0.5,-0.5])
    b= 0.7
    w=np.sum(x*y)+b
    if w<=0:
        return 0
    else:
        return 1

#XOR 异或门
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

    
print(AND(1,0))
print(OR(1,0))
print(NAND(1,0))
print(XOR(1,0))

