import random
import csv
from tabulate import tabulate
from Trainer import Trainer

class Classifier:
    def __init__(self, filename, preprocessor, training_test_ratio=0.7):
        self.spam = []
        self.ham = []

        self.traning_test_ration = training_test_ratio

        self.preprocess = preprocessor

        self.training_data = []
        self.test_data = []

        self.init_training_test_data(filename)

        self.trainer = Trainer(self.training_data, preprocessor)

    def init_training_test_data(self, filename):
        ratio = self.traning_test_ration
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            for row in reader:
                if int(row[-1]) is 1:
                    self.spam += [row[0]]
                else:
                    self.ham += [row[0]]

            for text in self.spam:
                if random.uniform(0, 1) < ratio:
                    self.training_data += [[text, 1]]
                else:
                    self.test_data += [[text, 1]]

            for text in self.ham:
                if random.uniform(0, 1) < ratio:
                    self.training_data += [[text, 0]]
                else:
                    self.test_data += [[text, 0]]

    def classify(self, message):
        res_prob_spam = 1
        res_prob_ham = 1

        message = self.preprocess.run(message)

        for word in message:
            prob_spam, prob_ham = self.trainer.get_probability(word)

            res_prob_spam *= prob_spam
            res_prob_ham *= prob_ham

        return int(res_prob_spam >= res_prob_ham)

    def calc_prediction_accuracy(self):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        for data in self.test_data:
            label = data[-1]
            message = data[0]

            prediction = self.classify(message)

            true_pos += int(label is 1 and prediction is 1)
            true_neg += int(label is 0 and prediction is 0)
            false_pos += int(label is 0 and prediction is 1)
            false_neg += int(label is 1 and prediction is 0)

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        print('')
        print("Akurasi: ", accuracy)

    def print_prediction(self, text_length):
        test_list = []
        for data in self.test_data:
            message = data[0][:text_length]
            label = 'SPAM' if data[-1] is 1 else 'HAM'

            prediction = 'SPAM' if self.classify(message) is 1 else 'HAM'

            test_list += [[message, prediction, label]]

        print(tabulate(test_list, headers=['Pesan', 'Prediksi', 'Target'], tablefmt='orgtbl'))

    def classify_message(self, message):
        label = int(self.classify(message))
        print('Message : {}'.format(message))
        print('Label   : {}'.format('SPAM' if label is 1 else 'Bukan Spam'))
