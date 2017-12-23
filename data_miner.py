import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report



def frame_gen(data1, data2, data3):
    col = ['id', 'text', 'sentiment', 'intensity']
    data1_frame = pd.read_csv(data1, sep='\t', names=col)
    data2_frame = pd.read_csv(data2, sep='\t', names=col)
    data3_frame = pd.read_csv(data3, sep='\t', names=col)
    frames = [data1_frame, data2_frame, data3_frame]
    final_frame = pd.concat(frames)
    return final_frame


def preprocess(line):
    str = line.split()
    no_punc = [char for char in str if char not in string.punctuation]
    no_punc = " ".join(no_punc)
    no_stop = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    return no_stop


anger_data = frame_gen('datasets/anger1.txt', 'datasets/anger2.txt', 'datasets/anger3.txt')
fear_data = frame_gen('datasets/fear1.txt', 'datasets/fear2.txt', 'datasets/fear3.txt')
joy_data = frame_gen('datasets/joy1.txt', 'datasets/joy2.txt', 'datasets/joy3.txt')
sadness_data = frame_gen('datasets/sadness1.txt', 'datasets/sadness2.txt', 'datasets/sadness3.txt')

all_list = [anger_data, fear_data, joy_data, sadness_data]
combined_frame = pd.concat(all_list)

X = combined_frame['text']
y = combined_frame['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

pipeline = Pipeline(
    [
        ('count_vector', CountVectorizer(analyzer=preprocess)),
        ('tfidf', TfidfTransformer()),
        ('model', MultinomialNB()),
    ]
)


if __name__ == '__main__':
    print("---Training Model---")
    model = pipeline.fit(X_train, y_train)
    print("---Testing Model---")
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
