import nltk
import io
import numpy as np
import os
from collections import Counter

END = 0
UNK = 1

class TrainingData:
    raw_text = []
    dictionary = {}
    X = []
    Y = []

    def load_input_data(self, file):
        with io.open(file, 'r') as f:
            for line in f:
                if (line != "\n"):
                    self.raw_text.append(line)
        f.close()

    def create_dictionary(self):
        tokenized = []
        words = []
        dictionary = {}
        print("Creating dictionary...")
        for sentence in self.raw_text:
            tokenized.append(nltk.wordpunct_tokenize(sentence))

        for sentence in tokenized:
            for word in sentence:
                words.append(word)

        if(len(words) > 10000):
            counter = Counter(words)
            mostCommonWords = counter.most_common(10000)
            words = [0] * len(mostCommonWords)
            for i in range(len(mostCommonWords)):
                words[i] = mostCommonWords[i][0]
            print(len(words))
        else:
            words = sorted(set(words))

        dictionary["<END>"] = END
        dictionary["<UNK>"] = UNK
        for i in range(0, len(words)):
            dictionary[words[i]] = i + 2
        self.dictionary = dictionary

    def prepare_training_data(self):
        print("Preparing training data...")

        sentence_number = 0
        for i in range(0, len(self.raw_text), 2):
            self.X.append([])
            sentence = []
            words = nltk.wordpunct_tokenize(self.raw_text[i])
            for word in words:
                sentence.append(word)
            sentence.append("<END>")

            for word in sentence:
                word_representation = np.zeros(shape=(1, len(self.dictionary)))
                try:
                    word_representation[0, self.dictionary[word]] = 1
                except KeyError:
                    word_representation[0, self.dictionary["<UNK>"]] = 1

                self.X[sentence_number].append(word_representation[0])
            sentence_number += 1

        sentence_number = 0
        for i in range(1, len(self.raw_text), 2):
            self.Y.append([])
            sentence = []
            words = nltk.wordpunct_tokenize(self.raw_text[i])
            for word in words:
                sentence.append(word)
            sentence.append("<END>")

            for word in sentence:
                word_representation = np.zeros(shape=(1, len(self.dictionary)))
                word_representation[0, self.dictionary[word]] = 1
                self.Y[sentence_number].append(word_representation[0])
            sentence_number += 1

    def load_training_data(self, file):
        self.load_input_data(file)
        self.create_dictionary()
        self.prepare_training_data()
        print("Training data is ready!")

    def loadOpenSubtitles(self):
        for filename in os.listdir('data'):
            f = open("data\\"+filename, 'r')
            for line in f:
                self.raw_text.append(line)
            f.close()

        self.create_dictionary()
        self.prepare_training_data()
        print("Training data is ready!")