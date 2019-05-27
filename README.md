# spam-sms-classifier
Python 3.7 SMS Spam Classifier using Naive Bayes Classifier and TF-IDF Vectorizer

### Depedencies
1. tabulate (Mencetak tabular data pada console) `pip install tabulate`
2. PySastrawi (Word Stemmer untuk Bahasa Indonesia) `pip install PySastrawi`

### Penggunaan
#### Memprediksi suatu pesan
`python main.py -m "masukkan pesan anda disini"`
#### Cetak hasil prediksi terhadap test data dengan panjang pesan sebanyak 50 karakter (default 100 karakter)
`python main.py -t -l 50`
#### Menyimpan file csv dari fitur kata dan data terpreproses dari dataset training
`python main.py -p`

### Sources
- [Dataset](http://nlp.yuliadi.pro/dataset)
- [Stopword Bahasa Indonesia](https://www.kaggle.com/oswinrh/indonesian-stoplist)
- [Alay Dictionary](https://github.com/AdrianAdyatma/Twitter-Sentiment-Analysis/blob/master/references/alay_dict.txt)
