import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import populate as p
import pickle
from friend_app.models import Sentiment


def dataset_access():
    data = list(Sentiment.objects.all())
    data_dict = {'text': [], 'category': [], 'intensity': []}
    for i in range(len(data)):
        data_dict['text'].append(data[i].text)
        data_dict['category'].append(data[i].category)
        data_dict['intensity'].append(data[i].intensity)
    df = pd.DataFrame(data_dict)
    X = df['text']
    z = df['intensity'].apply(lambda x: round(x,1))
    y = df['category']
    return (X, y, z)


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

    def train_cmodel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        cmodel = self.pipeline.fit(X_train, y_train)
        cmodel_filename = 'cmodel.pkl'
        cmodel_pkl = open(cmodel_filename, 'wb')
        pickle.dump(cmodel,cmodel_pkl)
        cmodel_pkl.close()
        predictions = cmodel.predict(X_test)
        report = classification_report(y_test, predictions)
        return report


    def train_imodel(self):
        self.y = self.y.apply(lambda x: round(x,1)).apply(str)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        imodel = self.pipeline.fit(X_train, y_train)
        imodel_filename = 'imodel.pkl'
        imodel_pkl = open(imodel_filename, 'wb')
        pickle.dump(imodel,imodel_pkl)
        imodel_pkl.close()
        predictions = imodel.predict(X_test)
        report = classification_report(y_test, predictions)
        return report



if __name__ == '__main__':
    print("---Training Model---")
    # data  = p.combined_frame
    # X = data['text']
    # y = data['sentiment']
    # z = data['intensity']
    X, y, z = dataset_access()
    category_predictor  = Predictor(X=X,y=y)
    creport = category_predictor.train_cmodel()

    intensity_predictor = Predictor(X=X, y=z)
    ireport = intensity_predictor.train_imodel()

    print("Classification report for category model is: {}".format(creport))
    print("Classification report for intensity model is: {}".format(ireport))
