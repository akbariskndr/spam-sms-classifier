import csv
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessor:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stopwords = self.initialize_stopwords()
        self.formalization_rules = self.initialize_formalization_rules()

    def remove_comma(infile, outfile):
        with open(infile, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            with open(outfile, mode='w', encoding='utf-8', newline='') as f_new:
                for row in reader:
                    writer = csv.writer(f_new, delimiter=',', quotechar='"')

                    identity = str.maketrans('', '', ';"')
                    text = ' '.join(row[:-1])
                    
                    is_spam = 0 if int(row[-1].translate(identity)) is 0 else 1
                    is_spam = str(is_spam)
                    
                    writer.writerow([text, is_spam])

    def initialize_stopwords(self):
        stopwords = []
        with open('./src/stopwords.csv', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            stopwords = [row[-1] for row in reader]

        return stopwords

    def initialize_formalization_rules(self):
        with open('./src/alay_dict.txt', encoding='utf-8') as f:
            words = {x.strip('\n').split(':')[0] : x.strip('\n').split(':')[1] for x in f}

        return words

    def remove_non_alphanumeric(self, text):
        return re.sub('[^A-z ]', ' ', text)

    def remove_excess_spaces(self, text):
        text = text.strip()
        return re.sub(' +', ' ', text)

    def tokenize(self, text):
        text = self.remove_non_alphanumeric(text)
        text = self.remove_excess_spaces(text)

        return text

    def case_fold(self, text):
        return text.lower()

    def remove_stop_words(self, text):
        words = text.split(' ')
        deleted_words = []
        for i in range(len(words)):
            if words[i] in self.stopwords:
                deleted_words.append(words[i])

        for i in deleted_words:
            words.remove(i)

        return " ".join(words)

    def stem(self, text):
        return self.stemmer.stem(text)

    def formalize(self, text):
        words = text.split(' ')
        for i in range(len(words)):
            if words[i] in self.formalization_rules:
                words[i] = self.formalization_rules[words[i]]

        return " ".join(words)

    def run(self, text):
        text = self.tokenize(text)
        text = self.case_fold(text)
        text = self.formalize(text)
        text = self.remove_stop_words(text)
        text = self.stem(text)
        text = [ x for x in text.split(' ') if len(x) > 2 ]
        return text
