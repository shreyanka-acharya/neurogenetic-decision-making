import numpy as np
from neural_network import NeuralNetwork

class Agent:
    def __init__(self):
        self.nn = NeuralNetwork(input_size=8, output_size=4)

    def act(self, state):
        # 10% chance to take random action
        if np.random.rand() < 0.1:
            return np.random.randint(0, 4)
        return np.argmax(self.nn.forward(state))

    def get_weights(self):
        return self.nn.get_weights()

    def set_weights(self, weights):
        self.nn.set_weights(weights)
