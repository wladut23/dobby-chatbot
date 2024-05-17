import io
import numpy as np
from scipy.special import expit
from datetime import datetime
from SoftmaxLayer import SoftmaxLayer

class Decoder:


    def __init__(self, vocabulary_length, no_of_cells, intermediate_layer=False, learning_rate=0.05):

        self.no_of_cells = no_of_cells
        self.vocabulary_length = vocabulary_length
        self.intermediate_layer = intermediate_layer
        if(not intermediate_layer):
            self.softmax_layer = SoftmaxLayer(vocabulary_length=27, no_of_cells=500)
            self.p = []


        self.W_C = np.random.uniform(-0.08, 0.08, (vocabulary_length, no_of_cells))
        self.W_f = np.random.uniform(-0.08, 0.08, (vocabulary_length, no_of_cells))
        self.W_i = np.random.uniform(-0.08, 0.08, (vocabulary_length, no_of_cells))
        self.W_o = np.random.uniform(-0.08, 0.08, (vocabulary_length, no_of_cells))

        self.U_C = np.random.uniform(-0.08, 0.08, (no_of_cells, no_of_cells))
        self.U_f = np.random.uniform(-0.08, 0.08, (no_of_cells, no_of_cells))
        self.U_i = np.random.uniform(-0.08, 0.08, (no_of_cells, no_of_cells))
        self.U_o = np.random.uniform(-0.08, 0.08, (no_of_cells, no_of_cells))

        self.bias_C = 0.05
        self.bias_f = 0.05
        self.bias_i = 0.05
        self.bias_o = 0.05
        self.learning_rate = learning_rate

        self.Cell_state = np.zeros((no_of_cells), dtype='float64')
        self.previous_Cell_state = np.zeros((no_of_cells), dtype='float64')
        self.h_state = np.zeros((no_of_cells), dtype='float64')
        self.previous_h_state = np.zeros((no_of_cells), dtype='float64')

        self.h_states = []
        self.Cell_states = []

        self.forget = []
        self.input = []
        self.output = []
        self.tanh_Ct = []

    def sigmoid(self, x):
        #normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return expit(x)

    def norm(self, d):
        suma = 0
        for element in d:
            suma += element ** 2
        s = np.sqrt(suma)
        return s

    def compute_loss(self, prediction, y):
        return -np.log(prediction[np.argmax(y)])

    def compute_partial_derivatives(self, p, x, y, o, tanh_Ct, f, i, t, derivative_L_wrt_Wprediction=np.array([0, 0]), derivative_L_wrt_h=1, derivative_L_wrt_X=1):
        if(not self.intermediate_layer):
            #compute the partial derivative of Loss function with respect to W_prediction
            derivative_L_wrt_Wprediction = np.outer((p-y), self.h_states[t])

            # compute the partial derivative of Loss function with respect to Wf and Uf
            derivative_L_wrt_h = derivative_L_wrt_h + np.dot(self.softmax_layer.W_prediction, (p - y))

        derivative_h_wrt_tanhCt = o
        derivative_tanhCt_wrt_ft = np.multiply((1-(tanh_Ct)**2), self.Cell_states[t-1])
        dsigmoid_f = np.multiply(f,(1-f))
        derivative_ft_wrt_Wf = np.outer(dsigmoid_f, x)
        derivative_ft_wrt_Uf = np.outer(dsigmoid_f, self.h_states[t-1])
        derivative_L_wrt_Wf = derivative_ft_wrt_Wf.T * (derivative_tanhCt_wrt_ft * derivative_h_wrt_tanhCt \
                                  * derivative_L_wrt_h)
        derivative_L_wrt_Uf = derivative_ft_wrt_Uf * (derivative_tanhCt_wrt_ft * derivative_h_wrt_tanhCt \
                                  * derivative_L_wrt_h)

        # compute the partial derivative of Loss function with respect to Wi and Ui
        derivative_tanhCt_wrt_it = np.multiply((1 - (tanh_Ct) ** 2), np.tanh(np.dot(self.W_C.T, x) + self.bias_C))
        dsigmoid_i = np.multiply(i, (1 - i))
        derivative_it_wrt_Wi = np.outer(dsigmoid_i, x)
        derivative_it_wrt_Ui = np.outer(dsigmoid_i, self.h_states[t-1])
        derivative_L_wrt_Wi = derivative_it_wrt_Wi.T * (derivative_tanhCt_wrt_it * derivative_h_wrt_tanhCt \
                                  * derivative_L_wrt_h)
        derivative_L_wrt_Ui = derivative_it_wrt_Ui * (derivative_tanhCt_wrt_it * derivative_h_wrt_tanhCt \
                                  * derivative_L_wrt_h)

        # compute the partial derivative of Loss function with respect to Wo and Uo
        dsigmoid_o = np.multiply(o, (1 - o))
        derivative_ot_wrt_Wo = np.outer(dsigmoid_o, x)
        derivative_ot_wrt_Uo = np.outer(dsigmoid_o, self.h_states[t-1])
        derivative_h_wrt_o = tanh_Ct
        derivative_L_wrt_Wo = derivative_ot_wrt_Wo.T * (derivative_h_wrt_o * derivative_L_wrt_h)
        derivative_L_wrt_Uo = derivative_ot_wrt_Uo * (derivative_h_wrt_o * derivative_L_wrt_h)

        # compute the partial derivative of Loss function with respect to Wc and Uc
        derivative_tanhCt_wrt_Candidates_t = np.multiply((1-(tanh_Ct)**2), i)
        dtanh = (1 - (tanh_Ct)**2)

        derivative_Candidates_t_wrt_Wc = np.outer(dtanh, x)
        derivative_Candidates_t_wrt_Uc = np.outer(dtanh, self.h_states[t-1])
        derivative_L_wrt_Wc = derivative_Candidates_t_wrt_Wc.T * (derivative_tanhCt_wrt_Candidates_t \
                                  * derivative_h_wrt_tanhCt * derivative_L_wrt_h)
        derivative_L_wrt_Uc = derivative_Candidates_t_wrt_Uc * derivative_tanhCt_wrt_Candidates_t \
                                  * derivative_h_wrt_tanhCt * derivative_L_wrt_h

        if(not self.intermediate_layer):
            # compute the partial derivative of Loss function with respect to previous_h
            derivative_L_wrt_hf = np.dot(self.U_f,
                                         derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_ft \
                                         * dsigmoid_f)
            derivative_L_wrt_hi = np.dot(self.U_i,
                                         derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_it \
                                         * dsigmoid_i)
            derivative_L_wrt_ho = np.dot(self.U_o, derivative_L_wrt_h * derivative_h_wrt_o * dsigmoid_o)
            derivative_L_wrt_hCandidates = np.dot(self.U_C, derivative_L_wrt_h * derivative_h_wrt_tanhCt \
                                                  * derivative_tanhCt_wrt_Candidates_t * dtanh)
            derivative_L_wrt_H = derivative_L_wrt_hf + derivative_L_wrt_hi + derivative_L_wrt_ho + derivative_L_wrt_hCandidates

            # compute the partial derivative of Loss function with respect to x
            derivative_L_wrt_xf = np.dot(self.W_f, derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_ft \
                                         * dsigmoid_f)
            derivative_L_wrt_xi = np.dot(self.W_i, derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_it \
                                         * dsigmoid_i)
            derivative_L_wrt_xo = np.dot(self.W_o, derivative_L_wrt_h * derivative_h_wrt_o * dsigmoid_o)
            derivative_L_wrt_xCandidates = np.dot(self.W_C, derivative_L_wrt_h * derivative_h_wrt_tanhCt \
                                                  * derivative_tanhCt_wrt_Candidates_t * dtanh)
            derivative_L_wrt_X = derivative_L_wrt_xf + derivative_L_wrt_xi + derivative_L_wrt_xo + derivative_L_wrt_xCandidates
        else:
            # compute the partial derivative of Loss function with respect to previous_h
            derivative_L_wrt_hf = np.dot(self.U_f,
                                         derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_ft \
                                         * dsigmoid_f)
            derivative_L_wrt_hi = np.dot(self.U_i,
                                         derivative_L_wrt_h * derivative_h_wrt_tanhCt * derivative_tanhCt_wrt_it \
                                         * dsigmoid_i)
            derivative_L_wrt_ho = np.dot(self.U_o, derivative_L_wrt_h * derivative_h_wrt_o * dsigmoid_o)
            derivative_L_wrt_hCandidates = np.dot(self.U_C, derivative_L_wrt_h * derivative_h_wrt_tanhCt \
                                                  * derivative_tanhCt_wrt_Candidates_t * dtanh)
            derivative_L_wrt_H = derivative_L_wrt_h + derivative_L_wrt_hf + derivative_L_wrt_hi + derivative_L_wrt_ho + derivative_L_wrt_hCandidates

        return derivative_L_wrt_Wf, derivative_L_wrt_Wi, derivative_L_wrt_Wo, derivative_L_wrt_Wc, \
               derivative_L_wrt_Uf, derivative_L_wrt_Ui, derivative_L_wrt_Uo, derivative_L_wrt_Uc, \
               derivative_L_wrt_Wprediction.T, derivative_L_wrt_H, derivative_L_wrt_X

    def tan(self, x):
        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return np.tanh(normalized_x)

    def softmax(self, x):
        #normalized_x = (x-np.min(x))/(np.max(x) - np.min(x))
        e = np.exp(x)
        sum = np.sum(e)
        return e/sum

    def compute_forward_pass(self, h, x):
        f = self.sigmoid(np.dot(self.W_f.T, x) + np.dot(self.U_f, h) + self.bias_f)
        i = self.sigmoid(np.dot(self.W_i.T, x) + np.dot(self.U_i, h) + self.bias_i)
        candidates = np.tanh(np.dot(self.W_C.T, x) + np.dot(self.U_C, h) + self.bias_C)
        o = self.sigmoid(np.dot(self.W_o.T, x) + np.dot(self.U_o, h) + self.bias_o)

        g = np.multiply(i, candidates)
        self.Cell_state = np.multiply(f, self.Cell_state) + g
        self.h_state = np.multiply(o, np.tanh(self.Cell_state))
        return self.h_state, f, i, o, candidates

    def backward_pass(self, dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_p=0):
        self.W_f -= self.learning_rate * dW_f
        self.W_i -= self.learning_rate * dW_i
        self.W_o -= self.learning_rate * dW_o
        self.W_C -= self.learning_rate * dW_C

        self.U_f -= self.learning_rate * dU_f
        self.U_i -= self.learning_rate * dU_i
        self.U_o -= self.learning_rate * dU_o
        self.U_C -= self.learning_rate * dU_C

        if(not self.intermediate_layer):
            self.softmax_layer.W_prediction -= self.learning_rate * dW_p

    def generate_sequence(self, h):
        output = ""
        for n in h:
            output = output + self.find_closest_prediction(n) + " "
        return output


    def find_closest_prediction(self, number):
        minimum = 1000
        for word, id in self.dictionary.items():
            error = abs(number - id)
            if(error < minimum):
                w = word
                minimum = error
        return w


    def save_parameters(self, file):
        with io.open(file, 'w') as f:
            for i in range (len(self.W_i)):
                for j in range (len(self.W_i[0])):
                    f.write(str(self.W_i[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.W_o)):
                for j in range (len(self.W_o[0])):
                    f.write(str(self.W_o[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.W_f)):
                for j in range (len(self.W_f[0])):
                    f.write(str(self.W_f[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.W_C)):
                for j in range (len(self.W_C[0])):
                    f.write(str(self.W_C[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.U_i)):
                for j in range (len(self.U_i[0])):
                    f.write(str(self.U_i[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.U_o)):
                for j in range (len(self.U_o[0])):
                    f.write(str(self.U_o[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.U_f)):
                for j in range (len(self.U_f[0])):
                    f.write(str(self.U_f[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            for i in range (len(self.U_C)):
                for j in range (len(self.U_C[0])):
                    f.write(str(self.U_C[i][j]) + " ")
                f.write("\n")
            f.write("\n")
            if(not self.intermediate_layer):
                for i in range (len(self.softmax_layer.W_prediction)):
                    for j in range (len(self.softmax_layer.W_prediction[0])):
                        f.write(str(self.softmax_layer.W_prediction[i][j]) + " ")
                    f.write("\n")
                f.write("\n")
        f.close()
        print("Done")

    def load_parameters(self, file):
        with io.open(file, 'r') as f:
            i=0
            while(i < len(self.W_i)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.W_i[i, j] = p[j]
                    i += 1
            i = 0
            while (i < len(self.W_o)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.W_o[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.W_f)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.W_f[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.W_C)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.W_C[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.U_i)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.U_i[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.U_o)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.U_o[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.U_f)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.U_f[i, j] = p[j]
                    i += 1

            i = 0
            while (i < len(self.U_C)):
                line = f.readline()
                if (line != '\n' and line != None):
                    p = np.fromstring(line, dtype='float64', sep=' ')
                    for j in range(len(p)):
                        self.U_C[i, j] = p[j]
                    i += 1

        f.close()

    def reset_values(self):
        self.Cell_state = np.zeros((self.no_of_cells), dtype='float64')
        self.h_state = np.zeros((self.no_of_cells), dtype='float64')
        self.Cell_states = []
        self.h_states = []
        self.forget = []
        self.input = []
        self.output = []
        self.tanh_Ct = []
        if(not self.intermediate_layer):
            self.p = []

    def forward_step(self, previous_h, x):
        if(self.intermediate_layer):
            h, forget, input, output, tanh_Ct = self.compute_forward_pass(previous_h, x)
            self.previous_h_state = h
            self.Cell_states.append(self.Cell_state)
            self.h_states.append(h)
            self.forget.append(forget)
            self.input.append(input)
            self.output.append(output)
            self.tanh_Ct.append(tanh_Ct)
            return h
        else:
            h, forget, input, output, tanh_Ct = self.compute_forward_pass(previous_h, x)
            self.previous_h_state = h
            self.Cell_states.append(self.Cell_state)
            self.h_states.append(h)
            self.forget.append(forget)
            self.input.append(input)
            self.output.append(output)
            self.tanh_Ct.append(tanh_Ct)

            prediction = self.softmax_layer.forward_step(h)
            self.p.append(prediction)
            return prediction

    def backprop(self, x, y, d_h=0):
        delta_Wf = 0
        delta_Wi = 0
        delta_Wo = 0
        delta_Wc = 0
        delta_Uf = 0
        delta_Ui = 0
        delta_Uo = 0
        delta_Uc = 0
        if(not self.intermediate_layer):
            loss = 0
            delta_Wp = 0
            delta_h = 0
            delta_x =  []
            d_h = 0
            for t in range(len(y)-1, -1, -1):
                loss += self.compute_loss(self.p[t], y[t])
                dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_p, d_h, d_x =  self.compute_partial_derivatives\
                    (self.p[t], x[t], y[t], self.output[t], self.tanh_Ct[t], self.forget[t], self.input[t], t, derivative_L_wrt_h=d_h)
                delta_Wf += dW_f
                delta_Wi += dW_i
                delta_Wo += dW_o
                delta_Wc += dW_C
                delta_Uf += dU_f
                delta_Ui += dU_i
                delta_Uo += dU_o
                delta_Uc += dU_C
                delta_Wp += dW_p
                delta_h += d_h
                delta_x.append(d_x)
            self.backward_pass(delta_Wf, delta_Wi, delta_Wo, delta_Wc, delta_Uf, delta_Ui, delta_Uo, delta_Uc, delta_Wp)
            s = self.norm(delta_h)
            if (s > 5):
                delta_h = (5 * delta_h / s)
            return loss, delta_x[::-1], delta_h
        else:
            d_prev_h = 0
            delta_prev_h = 0
            for t in range(len(y)-1, -1, -1):
                d_h[t] = d_h[t] + d_prev_h
                dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_p, d_prev_h, _ =  self.compute_partial_derivatives\
                    (None, x[t-1], y[t], self.output[t], self.tanh_Ct[t], self.forget[t], self.input[t], t, derivative_L_wrt_h=d_h[t])
                delta_Wf += dW_f
                delta_Wi += dW_i
                delta_Wo += dW_o
                delta_Wc += dW_C
                delta_Uf += dU_f
                delta_Ui += dU_i
                delta_Uo += dU_o
                delta_Uc += dU_C
                delta_prev_h += d_prev_h
                s = self.norm(delta_prev_h)
                if (s > 5):
                    delta_prev_h = (5 * delta_prev_h) / s
            self.backward_pass(delta_Wf, delta_Wi, delta_Wo, delta_Wc, delta_Uf, delta_Ui, delta_Uo, delta_Uc)
            return delta_prev_h


        '''
        self.previous_h_state = embedding
        self.Cell_state = np.zeros((self.no_of_cells), dtype='float64')
        dW_embedding_final = np.zeros((self.no_of_cells), dtype='float64')
        p = []
        x = np.zeros((self.vocabulary_length), dtype='float64')
        x[0] = 1
        loss = 0
        for i in range(len(y)):
            h, forget, input, output, tanh_Ct = self.compute_forward_pass(self.previous_h_state, x)
            prediction = self.softmax(np.dot(self.W_prediction.T, h))
            loss += self.compute_loss(prediction, y[i])
            dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_prediction, dW_embedding = self.compute_partial_derivatives(prediction,
                                                                                    x, y[i], output,
                                                                                    tanh_Ct, forget, input)
            dW_embedding_final += dW_embedding
            self.backward_pass(dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_prediction)
            self.previous_Cell_state = self.Cell_state
            self.previous_h_state = h
            x = prediction
            p.append(prediction)
        #dW_embedding_final = dW_embedding_final/len(y)
        suma = 0
        for element in dW_embedding_final:
            suma += element**2
        s = np.sqrt(suma)
        if(s > 5):
            dW_embedding_final = (5 * dW_embedding_final)/s
        return dW_embedding_final, p, loss
        '''

    def generate_sentence(self, embedded_text):
        self.previous_h_state = embedded_text
        self.Cell_state = np.zeros((self.no_of_cells), dtype='float64')
        x = np.zeros((self.vocabulary_length), dtype='float64')
        p = []
        prediction = [0, 1]
        while np.argmax(prediction) != 0:
            h, _, _, _, _ = self.compute_forward_pass(self.previous_h_state, x)
            prediction = self.softmax(np.dot(self.W_prediction.T, h))
            p.append(prediction)
            self.previous_Cell_state = self.Cell_state
            self.previous_h_state = h
            x = prediction
        return p[:-1]




    def train(self):
        print("Training has started...")
        epochs = 0
        old_loss = 100
        while True:
            #begin = datetime.now()
            for i in range(0, len(self.X)):
                self.Cell_state = np.zeros((self.vocabulary_length), dtype='float64')
                self.previous_h_state = np.zeros((self.vocabulary_length), dtype='float64')
                output = np.zeros((self.no_of_cells, self.vocabulary_length), dtype='float64')
                forget = np.ones((self.no_of_cells, self.vocabulary_length), dtype='float64')
                tanh_Ct = np.zeros((self.no_of_cells, self.vocabulary_length), dtype='float64')
                for j in range(len(self.Y[i])):
                    prediction, forget, input, output, tanh_Ct = self.compute_forward_pass(self.previous_h_state, self.X[i][0])
                    y_hat = self.softmax(np.dot(prediction, self.W_prediction))
                    if(epochs %100 == 0 and i==0):
                        loss = self.compute_loss(y_hat, self.Y[i][j])
                        #if(loss > old_loss):
                            #self.learning_rate = self.learning_rate*0.5
                            #print("Learning rate has been changed to ", self.learning_rate)
                        #old_loss = loss
                        print("loss:")
                        print(loss)
                        print("Expected: ")
                        print(self.Y[i][j])
                        print("PREDICTED: ")
                        print(y_hat)
                        print(epochs)
                    dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_prediction = self.compute_partial_derivatives(y_hat, self.X[i][0], self.Y[i][j], output, tanh_Ct, forget, input)
                    self.backward_pass(dW_f, dW_i, dW_o, dW_C, dU_f, dU_i, dU_o, dU_C, dW_prediction)
                    self.previous_Cell_state = self.Cell_state
                    #self.previous_h_state = prediction
                    self.previous_h_state = self.Y[i][j]
            #end = datetime.now()
            #print(end-begin)
            epochs +=1