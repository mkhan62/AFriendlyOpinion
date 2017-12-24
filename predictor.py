import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import populate as p


class Predictor:

    def preprocess(line):
        str = line.split()
        no_punc = [char for char in str if char not in string.punctuation]
        no_punc = " ".join(no_punc)
        no_stop = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
        return no_stop

    pipeline = Pipeline(
        [
            ('count_vector', CountVectorizer(analyzer=preprocess)),
            # ('tfidf', TfidfTransformer()),
            ('model', MultinomialNB()),
        ]
    )


    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        model = self.pipeline.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        return report

    def predict_category(self, text):
        model = self.pipeline.fit(self.X, self.y)
        return model.predict(text)

    def predict_intensity(self, text):
        self.y = self.y.apply(lambda x: round(x,1)).apply(str)
        model = self.pipeline.fit(self.X, self.y)
        prediction = model.predict(text)
        return prediction



if __name__ == '__main__':
    print("---Training Model---")
    data  = p.combined_frame
    X = data['text']
    y = data['sentiment']
    predictor  = Predictor(X=X,y=y)
    report = predictor.train_model()
    print(report)
