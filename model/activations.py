import numpy as np

def Softmax(x, axis = -1):
    print("softmax", x.ndim)
    x = x
    e_x = np.exp(x - np.max(x, axis, keepdims=True))

    softmax = e_x / np.sum(e_x, axis, keepdims=True)
    return softmax


def Sigmoid(x):
    print("Sigmoid", x.ndim)
    return 1 / (1 + np.exp(-x))


def SiLU(x):
    print("SiLU", x.ndim)
    return x * (1 / (1 + np.exp(-x)))


def SwiGLU(x, V):
    return x * (1 / (1 + np.exp(-x))) * V
