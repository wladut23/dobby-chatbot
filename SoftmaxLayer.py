import numpy as np


class SoftmaxLayer:

    def __init__(self, vocabulary_length, no_of_cells):
        self.vocabulary_length = vocabulary_length
        self. no_of_cells = no_of_cells

        self.W_prediction = np.random.uniform(-0.08, 0.08, (no_of_cells, vocabulary_length))

    def softmax(self, x):
        e = np.exp(x)
        sum = np.sum(e)
        return e / sum

    def forward_step(self, h):
        prediction = self.softmax(np.dot(self.W_prediction.T, h))
        return prediction
