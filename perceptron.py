import numpy as np
import sigmoid as s

class Perceptron():
    def __init__(self, inputlength, alpha=1):
        self.weights = np.random.randn(inputlength + 1)
        self.alpha = alpha

    def forward_step(self, input):
        self.biasedInput = np.append([1], input)
        output = self.biasedInput @ self.weights
        return s.sigmoid(output)

    def update(self, delta):
        self.weights -= self.alpha*delta*self.biasedInput

    def get_weights(self):
        return self.weights
