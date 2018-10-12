from dataset import Dataset
import ngram
import WK
import SSK
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='train')

print(news.data[0])

dataset = Dataset()

SVM_WK = WK.SVM("SVM_WK")
dT_WK = WK.DecisionTree("DT_WK")
kN_WK = WK.KNeighbors("5-NN_WK")
kN_WK_2 = WK.KNeighbors("3-NN_WK", neighbors = 3)
nb_WK = WK.NB("NB_WK")

SVM_5G = ngram.SVM("SVM_5G", ngram_size = 5)
dT_5G = ngram.DecisionTree("DT_5G", ngram_size = 5)
kN_5G = ngram.KNeighbors("5-NN_5G", ngram_size = 5)
kN_5G_2 = ngram.KNeighbors("3-NN_5G", neighbors = 3, ngram_size = 5)
nb_5G = ngram.NB("NB_5G", ngram_size = 5)

dataset.iterationStat(10, SVM_WK, dT_WK, kN_WK, kN_WK_2, nb_WK, SVM_5G, dT_5G, kN_5G, kN_5G_2, nb_5G)

"""SVM_SSK = SSK.SVM("SVM_SSK")

SVM_SSK.train(dataset)
Y = SVM_SSK.predict(dataset)
print(Y)
dataset.evaluatePrediction(Y)"""
