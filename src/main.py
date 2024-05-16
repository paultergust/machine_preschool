import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent, linear_regression, quadratic_regression

def plot_graph(xvalues, yvalues, lr):
    plt.scatter(xvalues, yvalues)
    plt.plot(xvalues, lr)
    plt.show()
    
if __name__ = '__main__':
    pass
