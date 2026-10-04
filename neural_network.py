import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=8, hidden_size=8, output_size=4):
        self.w1 = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b1 = np.random.uniform(-1, 1, (hidden_size,))
        self.w2 = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b2 = np.random.uniform(-1, 1, (output_size,))

    def forward(self, x):
        x = np.array(x)
        h = np.tanh(np.dot(self.w1, x) + self.b1)
        out = np.dot(self.w2, h) + self.b2
        return out

    def get_weights(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def set_weights(self, weights):
        self.w1, self.b1, self.w2, self.b2 = [np.array(w) for w in weights]
