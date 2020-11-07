import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.e**-x)
def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))

if(False):
    line = np.array([x for x in range(-10,10)])

    plt.plot(line,sigmoid(line))
    plt.plot(line,sigmoidprime(line))
    plt.show()
