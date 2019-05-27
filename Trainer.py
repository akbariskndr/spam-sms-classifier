import math
import csv

class Trainer:
    def __init__(self, training_data, preprocessor):
        self.preprocess = preprocessor
        self.training_data = training_data

        self.spam_dataset = 0
        self.ham_dataset = 0

        self.spam_words_count = 0
        self.ham_words_count = 0

        self.spam_word_list = []
        self.ham_word_list = []

        self.sum_tfidf = {
            "spam": 0,
            "ham": 0,
        }

        self.word_data = {}

        self.run()

    def run(self):
        with open('preprocessed.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            for row in self.training_data:
                message = self.preprocess.run(row[0])
                label = int(row[-1])

                self.count_occurence(message, label)

                message = ' '.join(message)
                writer.writerow([message, label])

        self.init_tf_idf()
        self.init_probability()

        self.save_words_to_csv('word_features.csv')

    def define_word_data(self, word, label):
        self.word_data[word] = {
            "occurence": {
                "spam": 1 if label is 1 else 0,
                "ham": 1 if label is 0 else 1,
            },
            "appear_in": {
                "spam": 0,
                "ham": 0,
            },
            "prob": {
                "spam": 0,
                "ham": 0,
            },
            "tfidf": {
                "spam": 0,
                "ham": 0,
            },
        }

    def count_occurence(self, message, label):
        word_list = []
        for word in message:
            if word in self.word_data.keys():
                if label is 1:
                    self.word_data[word]["occurence"]["spam"] += 1
                    self.spam_words_count += 1
                else:
                    self.word_data[word]["occurence"]["ham"] += 1
                    self.ham_words_count += 1
            else:
                self.define_word_data(word, label)

            if word not in word_list:
                word_list += [word]

        for word in word_list:
            if label is 1:
                self.word_data[word]["appear_in"]["spam"] += 1
                if word not in self.spam_word_list:
                    self.spam_word_list + [word]
            else:
                self.word_data[word]["appear_in"]["ham"] += 1
                if word not in self.ham_word_list:
                    self.ham_word_list + [word]
        
        if label is 1:
            self.spam_dataset += 1
        else:
            self.ham_dataset += 1

    def init_tf_idf(self):
        sum_tfidf_spam = 0
        sum_tfidf_ham = 0
        for word in self.word_data.keys():
            tf_spam = self.word_data[word]["occurence"]["spam"]
            idf_spam = math.log((self.spam_dataset + self.ham_dataset) / (self.word_data[word]["appear_in"]["spam"] + self.word_data[word]["appear_in"]["ham"]))
            tfidf_spam = tf_spam * idf_spam

            self.word_data[word]["tfidf"]["spam"] = tfidf_spam
            sum_tfidf_spam += tfidf_spam

            tf_ham = self.word_data[word]["occurence"]["ham"]
            idf_ham = math.log((self.spam_dataset + self.ham_dataset) / (self.word_data[word]["appear_in"]["spam"] + self.word_data[word]["appear_in"]["ham"]))
            tfidf_ham = tf_ham * idf_ham

            self.word_data[word]["tfidf"]["ham"] = tfidf_ham
            sum_tfidf_ham += tfidf_ham

        self.sum_tfidf["spam"] = sum_tfidf_spam
        self.sum_tfidf["ham"] = sum_tfidf_ham

    def init_probability(self):
        for word in self.word_data.keys():
            tfidf_spam = self.word_data[word]["tfidf"]["spam"]
            tfidf_ham = self.word_data[word]["tfidf"]["ham"]

            prob_spam, prob_ham = self.calc_probability(tfidf_spam, tfidf_ham)

            self.word_data[word]["prob"]["spam"] = prob_spam
            self.word_data[word]["prob"]["ham"] = prob_ham
    
    def calc_probability(self, tfidf_spam, tfidf_ham, alpha=1):
        spam_word_list_count = len(self.spam_word_list)
        ham_word_list_count = len(self.ham_word_list)

        sum_tfidf_spam = self.sum_tfidf["spam"]
        sum_tfidf_ham = self.sum_tfidf["ham"]
        prob_spam = (tfidf_spam + alpha) / (sum_tfidf_spam + (alpha * spam_word_list_count))
        prob_ham = (tfidf_ham + alpha) / (sum_tfidf_ham + (alpha * ham_word_list_count))

        return (prob_spam, prob_ham)

    def get_probability(self, word):
        prob_spam = 0
        prob_ham = 0
        if word in self.word_data:
            prob_spam = self.word_data[word]["prob"]["spam"]
            prob_ham = self.word_data[word]["prob"]["ham"]
        else:
            prob_spam, prob_ham = self.calc_probability(0, 0)

        return (prob_spam, prob_ham)

    def save_words_to_csv(self, filename):
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            writer.writerow([
                "Kata", "Jumlah Kemunculan di Dataset Spam",
                "Jumlah Kemunculan di Dataset Ham", "Jumlah Dokumen Spam",
                "Jumlah Dokumen Ham", "Probabilitas Spam", "Probabilitas Ham"
            ])
            for word in self.word_data.keys():
                spam_occurence = self.word_data[word]["occurence"]["spam"]
                ham_occurence = self.word_data[word]["occurence"]["ham"]

                appear_in_spam_docs = self.word_data[word]["appear_in"]["spam"]
                appear_in_ham_docs = self.word_data[word]["appear_in"]["ham"]

                prob_spam = self.word_data[word]["prob"]["spam"]
                prob_ham = self.word_data[word]["prob"]["ham"]
                writer.writerow([word, spam_occurence, ham_occurence, appear_in_spam_docs, appear_in_ham_docs, prob_spam, prob_ham])