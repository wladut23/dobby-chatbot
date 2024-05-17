import io
import numpy as np


class Encoder:

    X = []
    Y = []

    def __init__(self, vocabulary_length, encoding_length, intermediate_layer=False, learning_rate=0.05):
        self.vocabulary_length = vocabulary_length
        self.encoding_length = encoding_length
        self.intermediate_layer = intermediate_layer

        self.W = np.random.uniform(-0.08, 0.08, (vocabulary_length, encoding_length))
        self.U = np.random.uniform(-0.08, 0.08, (encoding_length, encoding_length))
        self.previous_h = np.zeros((encoding_length), dtype='float64')
        self.h_states = []

        self.learning_rate = learning_rate

    def create_customized_training_data(self, window):
        for sequence in self.sequences:
            for i in range(len(sequence)-window):
                count = 1
                while(count <= window):
                    self.X.append(sequence[i])
                    self.Y.append(sequence[i+count])
                    count+=1
            for i in range(len(sequence)-1, -1+window, -1):
                count = 1
                while (count <= window):
                    self.X.append(sequence[i])
                    self.Y.append(sequence[i - count])
                    count += 1

    def softmax(self, x):
        e = np.exp(x)
        sum = np.sum(e)
        return e/sum

    def norm(self, d):
        suma = 0
        for element in d:
            suma += element ** 2
        s = np.sqrt(suma)
        return s

    def cross_entropy_loss(self, prediction, expected):
        return -np.sum(np.dot(expected, np.log(prediction)))

    def save_parameters(self, file):
        with io.open(file, 'w') as f:
            for i in range (len(self.W)):
                for j in range (len(self.W[0])):
                    f.write(str(self.W[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.U)):
                for j in range (len(self.U[0])):
                    f.write(str(self.U[i][j]) + " ")
                f.write("\n")
            f.write("\n")

        f.close()

    def load_parameters(self, file):
        with io.open(file, 'r') as f:
            i=0
            while(i < len(self.W)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.W[i, j] = p[j]
                    i += 1

            i=0
            while (i < len(self.U)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.U[i, j] = p[j]
                    i += 1

        f.close()

    def train(self):
        W_xh = np.random.uniform(0, 0.5, (self.vocabulary_length, self.encoding_length))
        W_ho = np.random.uniform(0, 0.5, (self.encoding_length, self.vocabulary_length))
        window_size = 2
        self.create_customized_training_data(window_size)
        epochs = 0
        while epochs < 50:
            for i in range(len(self.X)):
                h = np.dot(self.X[i], W_xh)
                y = self.softmax(np.dot(h, W_ho))
                loss = self.cross_entropy_loss(y, self.Y[i])
                if(loss > 0.5):
                    dW_ho, dW_xh = self.compute_gradient(self.X[i], y, self.Y[i])
                    W_ho, W_xh = self.backpropagate(W_ho, dW_ho, W_xh, dW_xh)
                if(epochs % 50 == 0):
                    print(loss)
                    print("expected: ")
                    print(self.Y[i])
                    print("prediction: ")
                    print(y)

                    self.save_parameters("encoder_param.txt")
            epochs+=1

    def tan(self, x):
        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return np.tanh(normalized_x)

    def encode(self, sequence):
        self.previous_h = np.zeros((self.encoding_length), dtype='float64')
        for i in range(len(sequence) - 1):
            x = sequence[i]
            h = np.tanh((np.dot(self.W.T, x) + np.dot(self.U, self.previous_h)))
            self.previous_h = h
            self.h_states.append(h)
        self.h_states.append(np.zeros(len(h),dtype='float64'))
        return self.previous_h

    def backpropagate(self, sequence, d_h):
        if(self.intermediate_layer):
            dU = [0]
            dW = [0]
            delta_h1 = []
            for t in range(len(sequence)-1, -1, -1):
                x = sequence[t]
                dW += np.outer(d_h, (1 - self.h_states[t]**2) * x)

                dU += np.outer(d_h, (1 - self.h_states[t]**2) * self.h_states[t-1])
                delta_h1.append(np.dot(self.W, d_h * (1 - self.h_states[t]**2)))
                d_h = np.dot(self.U, d_h * (1 - self.h_states[t]**2))
                s = np.linalg.norm(d_h)
                if(s > 5):
                    d_h = (d_h * 5)/s

            s = np.linalg.norm(dW)
            if(s > 5):
                dW = (dW * 5) / s
            self.W -= dW.T * self.learning_rate

            s = np.linalg.norm(dU)
            if (s > 5):
                dU = (dU * 5) / s
            self.U -= dU * self.learning_rate
            self.h_states = []
            return delta_h1[::-1]
        else:
            dU = [0]
            dW = [0]
            dh_prev = 0
            for t in range(len(sequence) - 2, -1, -1):
                x = sequence[t]
                dH = d_h[t] + dh_prev
                dW += np.outer(dH * (1 - self.h_states[t] ** 2), x)
                dU += np.outer(dH * (1 - self.h_states[t] ** 2), self.h_states[t - 1])
                dh_prev = np.dot(self.U, dH * (1 - self.h_states[t]**2))
                s = self.norm(dh_prev)
                if(s > 5):
                    dh_prev = (5 * dh_prev) / s
                d_h[t-1] = d_h[t-1] + dh_prev

            s = np.linalg.norm(dW)
            if (s > 5):
                dW = (dW * 5) / s
            self.W -= dW.T * self.learning_rate

            s = np.linalg.norm(dU)
            if (s > 5):
                dU = (dU * 5) / s
            self.U -= dU * self.learning_rate
            self.h_states = []




