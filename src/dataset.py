import initDataset
import random as rd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
from nltk.corpus import stopwords

class Dataset:
    def __init__(self, labels = ["earn", "acq", "crude", "corn"], nbTrain = [152, 114, 76, 38], nbTest = [40, 25, 15, 10]):
        print("Preparing dataset")
        self.__trainSetTot, self.__testSetTot = initDataset.extractDataSet()
        for i in range(len(self.__trainSetTot)):
            self.__trainSetTot[i] = self.__trainSetTot[i][:10]
        self.__nbLabels = len(labels)
        self.__labels = labels
        self.__nbTrain = nbTrain
        self.__nbTest = nbTest

        self.__trainSet = []
        self.__testSet = []
        self.__trainLabels = []
        self.__testLabels = []

        self.sampleData()

        print("Dataset prepared")

    def getTrainSet(self):
        return self.__trainSet

    def getTestSet(self):
        return self.__testSet

    def getTrainLabels(self):
        return self.__trainLabels

    def getTestLabels(self):
        return self.__testLabels

    def evaluatePrediction(self, prediction):
        ok = 0.0
        tot = 0.0
        for i in range(len(self.__testLabels)):
            tot += 1
            if prediction[i] == self.__testLabels[i]:
                ok += 1

        print(ok, tot, ok/tot)

    def sampleData(self):
        self.__trainSet = []
        self.__testSet = []
        self.__trainLabels = []
        self.__testLabels = []
        trainSetTemp = []
        testSetTemp = []

        for i in range(self.__nbLabels):
            trainSetTemp.append([])
            testSetTemp.append([])

        for x in self.__trainSetTot:
            for j in range(len(x[1])):
                for i in range(self.__nbLabels):
                    if x[1][j] == self.__labels[i]:
                        trainSetTemp[i].append(x[0])

        for x in self.__testSetTot:
            for j in range(len(x[1])):
                for i in range(self.__nbLabels):
                    if x[1][j] == self.__labels[i]:
                        testSetTemp[i].append(x[0])

        for i in range(self.__nbLabels):
            trainSetTemp[i] = rd.sample(trainSetTemp[i], self.__nbTrain[i])
            testSetTemp[i] = rd.sample(testSetTemp[i], self.__nbTest[i])

        for i in range(self.__nbLabels):
            self.__trainLabels += [i]*len(trainSetTemp[i])
            self.__testLabels += [i]*len(testSetTemp[i])

            self.__trainSet += trainSetTemp[i]
            self.__testSet += testSetTemp[i]

        c = list(zip(self.__trainSet, self.__trainLabels))
        rd.shuffle(c)
        self.__trainSet, self.__trainLabels = zip(*c)

    def iterationStat(self, number, *args):
        F1 = []
        precision = []
        recall = []

        for i in range(len(args)):
            l = []
            ll = []
            lll = []
            for j in range(self.__nbLabels):
                l.append([])
                ll.append([])
                lll.append([])
            F1.append(l)
            recall.append(ll)
            precision.append(lll)

        for i in range(number): #for each iteration
            for j in range(len(args)): #for each model
                args[j].train(self)
                prediction = args[j].predict(self)

                F1_temp = f1_score(self.__testLabels, prediction, average=None)
                recall_temp = recall_score(self.__testLabels, prediction, average=None)
                precision_temp = precision_score(self.__testLabels, prediction, average=None)
                for k in range(self.__nbLabels):
                    F1[j][k].append(F1_temp[k])
                    recall[j][k].append(recall_temp[k])
                    precision[j][k].append(precision_temp[k])
            self.sampleData()

        for j in range(len(args)):
            for k in range(self.__nbLabels):
                F1_mean = np.mean(F1[j][k])
                F1_std = np.std(F1[j][k])
                F1[j][k] = [round(F1_mean, 3), round(F1_std, 3)]

                recall_mean = np.mean(recall[j][k])
                recall_std = np.std(recall[j][k])
                recall[j][k] = [round(recall_mean, 3), round(recall_std, 3)]

                precision_mean = np.mean(precision[j][k])
                precision_std = np.std(precision[j][k])
                precision[j][k] = [round(precision_mean, 3), round(precision_std, 3)]
        print("")
        print("Category\tModel\tF1\t\tPrecision\t\tRecall\t\t")
        print("\t\t\tMean\tSD\tMean\tSD\tMean\tSD")
        for i in range(self.__nbLabels):
            print(self.__labels[i])
            for k in range(len(args)):
                print("\t\t" + args[k].name)
                print("\t\t\t" + str(F1[k][i][0]) + "\t" + str(F1[k][i][1]) + "\t" + str(precision[k][i][0]) + "\t" + str(precision[k][i][1]) + "\t" + str(recall[k][i][0]) + "\t" + str(recall[k][i][1]))
