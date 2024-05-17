import nltk
import numpy as np
import io
from datetime import datetime
import threading
#from pynput.keyboard import Key, Controller
from TrainingData import TrainingData
from Encoder import Encoder
from Decoder import Decoder


data = TrainingData()
data.load_training_data(file='test2.txt')
# data.loadOpenSubtitles()

print(data.dictionary)
#listX = data.X[0]
#listY = data.Y[0]

print("----------------------------------")
vocabulary_length = len(data.dictionary)
print(vocabulary_length)

#rnn_encoder.create_customized_training_data(window=2)
#rnn_encoder.train()
#decoder_training_data = []
#for i in range(len(data.X)):
#    decoder_training_data.append(rnn_encoder.encode(data.X[i]))

#lstm_decoder = Decoder(300, vocabulary_length, decoder_training_data, data.Y)
#lstm_decoder.train()
encoder_layer1 = Encoder(vocabulary_length, encoding_length=500)
encoder_layer2 = Encoder(vocabulary_length=500, encoding_length=500, intermediate_layer=True)
decoder_layer1 = Decoder(vocabulary_length, no_of_cells=500, intermediate_layer=True)
decoder_layer2 = Decoder(vocabulary_length=500, no_of_cells=500)

def talk(p, y):
    predictie = ""
    label = ""
    for i in range(len(p)):
        for word, id in data.dictionary.items():
            if(np.argmax(p[i]) == id):
                predictie += word + " "
            if(np.argmax(y[i]) == id):
                label += word + " "
    print("Label: ", label)
    print("Predictie: ", predictie)

def talk1(p):
    predictie = ""
    for i in range(len(p)):
        for word, id in data.dictionary.items():
            if (np.argmax(p[i]) == id):
                predictie += word + " "
    print("ChatBot: ", predictie)

def train():
    epochs = 0
    losses = []
    N = 0
    contor = 0
    end_token = np.zeros(vocabulary_length)
    end_token[0] = 1
    while True:
        total_loss = 0
        for k in range(len(data.X)):
            layer1_embedding = encoder_layer1.encode(data.X[k])
            layer2_embedding = encoder_layer2.encode(encoder_layer1.h_states)
            h1 = decoder_layer1.forward_step(layer1_embedding, data.X[k][-1])
            prediction = decoder_layer2.forward_step(layer2_embedding, h1)
            for t in range(1, len(data.Y[k]), 1):
                h1 = decoder_layer1.forward_step(decoder_layer1.previous_h_state, prediction)
                prediction = decoder_layer2.forward_step(decoder_layer2.previous_h_state, h1)
            p = decoder_layer2.p
            p.append(end_token)

            decoder_layer1.Cell_states.append(np.zeros((decoder_layer1.no_of_cells), dtype='float64'))
            decoder_layer2.Cell_states.append(np.zeros((decoder_layer2.no_of_cells), dtype='float64'))
            decoder_layer1.h_states.append(layer1_embedding)
            decoder_layer2.h_states.append(layer2_embedding)

            loss, delta_h1, delta_encoder2 = decoder_layer2.backprop(decoder_layer1.h_states[:-1], data.Y[k])
            total_loss += loss
            N += len(data.Y[k])
            delta_encoder1 = decoder_layer1.backprop(p, data.Y[k], delta_h1)
            delta_layer2 = encoder_layer2.backpropagate(encoder_layer1.h_states[:-1], delta_encoder2)
            delta_layer2[-1] = delta_layer2[-1] + delta_encoder1
            suma = 0
            for d in range(len(delta_layer2)):
                for element in delta_layer2[d]:
                    suma += element ** 2
                s = np.sqrt(suma)
                if (s > 5):
                    delta_layer2[d] = (5 * delta_layer2[d]) / s
            encoder_layer1.backpropagate(data.X[k], delta_layer2)

            decoder_layer1.reset_values()
            decoder_layer2.reset_values()

            if (epochs % 10 == 0):
                print("Epochs: ", epochs)
                talk(p[:-1], data.Y[k])
                print("-------======================--------")
        losses.append(total_loss/N)
        print(losses[epochs])
        N = 0
        epochs += 1

        if((len(losses) > 1 and losses[-1] > losses[-2])):
            contor += 1
            if(contor > 1):
                decoder_layer1.learning_rate = decoder_layer1.learning_rate * 0.5
                decoder_layer2.learning_rate = decoder_layer2.learning_rate * 0.5
                encoder_layer1.learning_rate = encoder_layer1.learning_rate * 0.5
                encoder_layer2.learning_rate = encoder_layer2.learning_rate * 0.5
                print("Learning rate set to: ", decoder_layer1.learning_rate)
                contor = 0
        #if(epochs % 100 == 0):
            #encoder_layer1.save_parameters("encoder1_parameters.txt")
            #encoder_layer2.save_parameters("encoder2_parameters.txt")
            #decoder_layer1.save_parameters("decoder1_parameters.txt")
            #decoder_layer2.save_parameters("decoder2_parameters.txt")
        #keyboard = Controller()
        #if(keyboard.is_pressed('q')):
            #break

def load_parameters():
    encoder_layer1.load_parameters("encoder1_parameters.txt")
    encoder_layer2.load_parameters("encoder2_parameters.txt")
    decoder_layer1.load_parameters("decoder1_parameters.txt")
    decoder_layer2.load_parameters("encoder1_parameters.txt")


def start_chat():
    print("Chatbot connected!")
    while True:
        text = input()
        sequence = []
        for word in nltk.wordpunct_tokenize(text):
            word_representation = np.zeros(len(data.dictionary))
            try:
                word_representation[data.dictionary[word]] = 1
            except:
                word_representation[data.dictionary["<UNK>"]] = 1
            sequence.append(word_representation)
        end_token = np.zeros(len(data.dictionary))
        end_token[0] = 1
        layer1_embedding = encoder_layer1.encode(sequence)
        layer2_embedding = encoder_layer2.encode(encoder_layer1.h_states)
        number_of_words = 0
        h1 = decoder_layer1.forward_step(layer1_embedding, end_token)
        prediction = decoder_layer2.forward_step(layer2_embedding, h1)
        number_of_words += 1
        while np.argmax(prediction) != 0 and number_of_words < 25:
            h1 = decoder_layer1.forward_step(decoder_layer1.previous_h_state, prediction)
            prediction = decoder_layer2.forward_step(decoder_layer2.previous_h_state, h1)
            number_of_words += 1
        p = decoder_layer2.p
        talk1(p)

        decoder_layer1.reset_values()
        decoder_layer2.reset_values()
        encoder_layer1.h_states = []
        encoder_layer2.h_states = []


train()
start_chat()
