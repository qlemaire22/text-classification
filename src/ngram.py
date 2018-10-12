from model import Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


class SVM(Model):
    def __init__(self, name, ngram_size, max_iter=5):
        self.max_iter = max_iter
        self.ngram_size = ngram_size
        Model.__init__(self, name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(ngram_size,ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                    alpha=1e-3, random_state=42,
                                                  max_iter=self.max_iter, tol=None)),
        ])

    def train(self, dataset):
        print("Training model " + self.name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(self.ngram_size,self.ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                    alpha=1e-3, random_state=42,
                                                  max_iter=self.max_iter, tol=None)),
        ])
        self.text_clf.fit(dataset.getTrainSet(), dataset.getTrainLabels())
        print("Traning done")

    def predict(self, dataset):
        Y = self.text_clf.predict(dataset.getTestSet())
        return Y

class NB(Model):
    def __init__(self, name, ngram_size):
        Model.__init__(self, name)
        self.ngram_size = ngram_size

        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(ngram_size,ngram_size))),
                                ('clf', MultinomialNB()),
        ])

    def train(self, dataset):
        print("Training model " + self.name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(self.ngram_size,self.ngram_size))),
                                ('clf', MultinomialNB()),
        ])

        self.text_clf.fit(dataset.getTrainSet(), dataset.getTrainLabels())
        print("Traning done")

    def predict(self, dataset):
        Y = self.text_clf.predict(dataset.getTestSet())
        return Y


class KNeighbors(Model):
    def __init__(self, name, ngram_size, neighbors = 5):
        self.ngram_size = ngram_size
        self.neighbors = neighbors
        Model.__init__(self, name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(ngram_size,ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', KNeighborsClassifier(n_neighbors=self.neighbors)),
        ])

    def train(self, dataset):
        print("Training model " + self.name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(self.ngram_size,self.ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', KNeighborsClassifier(n_neighbors=self.neighbors)),
        ])

        self.text_clf.fit(dataset.getTrainSet(), dataset.getTrainLabels())
        print("Traning done")

    def predict(self, dataset):
        Y = self.text_clf.predict(dataset.getTestSet())
        return Y

class DecisionTree(Model):
    def __init__(self, name, ngram_size):
        Model.__init__(self, name)
        self.ngram_size = ngram_size
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(ngram_size,ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', DecisionTreeClassifier()),
        ])

    def train(self, dataset):
        print("Training model " + self.name)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(self.ngram_size,self.ngram_size))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', DecisionTreeClassifier()),
        ])

        self.text_clf.fit(dataset.getTrainSet(), dataset.getTrainLabels())
        print("Traning done")

    def predict(self, dataset):
        Y = self.text_clf.predict(dataset.getTestSet())
        return Y
