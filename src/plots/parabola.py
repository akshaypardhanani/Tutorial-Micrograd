import math
import numpy as np
import matplotlib.pyplot as plt
from src.scalar import f


def plot(xs: list, ys: list):
    plt.plot(xs, ys)
    plt.show()
    

if __name__ == '__main__':
    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)
    plot(xs, ys)
