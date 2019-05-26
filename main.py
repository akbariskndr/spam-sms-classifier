import argparse
from Classifier import Classifier
from Preprocessor import Preprocessor

preprocessor = Preprocessor()
classifier = Classifier('./src/dataset.csv', preprocessor, 0.7)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--message', type=str)
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-l', '--length', type=int, default=100)

args = parser.parse_args()

if args.message:
    classifier.classify_message(args.message)
elif args.test:
    classifier.print_prediction(args.length)
    classifier.calc_prediction_accuracy()