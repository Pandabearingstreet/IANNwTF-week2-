import numpy as np
import perceptron as p
import sigmoid as s

class MLP():
    def __init__(self, inputlength):
        self.hiddenlayer = [
            p.Perceptron(inputlength),
            p.Perceptron(inputlength),
            p.Perceptron(inputlength),
            p.Perceptron(inputlength)
        ]
        self.output = [
            p.Perceptron(4)
        ]

    def forward_step(self, input):
        self.input = input
        self.hiddenlayerActivation = [
            perceptron.forward_step(input) for perceptron in self.hiddenlayer
        ]
        self.prediction = self.output[0].forward_step(self.hiddenlayerActivation)

    def backprop_step(self, target):
        outputDelta = - (target - self.prediction) * s.sigmoidprime(np.append([1],self.hiddenlayerActivation) @ self.output[0].get_weights())
        self.output[0].update(outputDelta)
        self.hiddenlayerDeltas = [
            outputDelta * self.output[0].get_weights()[i] * s.sigmoidprime(np.append([1],self.input) @ self.hiddenlayer[i].get_weights())
            for i in range(4)]

    def get_prediction(self):
        return self.prediction
