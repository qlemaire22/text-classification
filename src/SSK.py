from model import Model
from sklearn import svm
import numpy as np
from functools import lru_cache
from collections import Counter
import csv
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(1500)

class SVM(Model):
    def __init__(self, name, dataset, n):
        Model.__init__(self, name)
        self.subseq_length = n
        self.lambda_decay = 0.5
        self.text_clf = svm.SVC(kernel='precomputed')
        self.alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                         't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        self.S = self.preprocessiong(dataset, n)
        self.n = n

    def computeNgram(self, n):
        list_ngram = self.alphabet[:]  # copy / 1gram
        for i in range(n - 1):
            newlist_ngram = list_ngram[:]  # copy
            for ngram in list_ngram:
                for char in self.alphabet:
                    newlist_ngram.append(ngram + char)
            list_ngram = newlist_ngram
        # print(list_ngram) # check already done
        return list_ngram

    def ngrams(self, text, n):
        if (len(text) < n):
            return []
        text_length = len(text)
        text_list = (text_length - n) * [0]
        text_list[0] = text[0:n]
        for i in range(1, text_length - n):
            text_list[i] = text_list[i - 1][1:] + text[i + n]
        return text_list

    def computeS(self, list_ngram):
        count = Counter(list_ngram)
        cuple = sorted(count.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in cuple[:200]]

    def preprocessiong(self, dataset, n):
        set_text = dataset.getTrainSet()[:100]  # 100 first, maybe we have to change and stock the 100 first from the original list.
        list_ngram = []
        for text in set_text:
            list_ngram += self.ngrams(text, n)

        return self.computeS(list_ngram)

    # FIRST TECHNIC
    @lru_cache(maxsize=None)
    def _K(self, n, sx, t):
        sum_cum = 0
        for s in self.S:
            sum_cum += self.__K(n, s, sx) * self.__K(n, s, t)
        return sum_cum

    @lru_cache(maxsize=None)
    def __K(self, n, sx, t):
        if (sx == " " or t == " "):
            return 0

        if (min(len(sx), len(t)) < n):
            return 0
        partial_sum = 0
        s, x = sx[:-1], sx[-1]
        for j in range(len(t)):
            if (t[j] == x):
                partial_sum += self._K1(n - 1, s, t[:j])
        return self.__K(n, s, t) + partial_sum * self.lambda_decay ** 2

    @lru_cache(maxsize=None)
    def _K1(self, i, sx, t):
        #print(i, sx, t)
        if (i == 0):
            return 1
        if (min(len(sx), len(t)) < i):
            return 0
        partial_sum = 0
        s, x = sx[:-1], sx[-1]
        len_t = len(t)

        return self.lambda_decay * self._K1(i, s, t) + self._K2(i, sx, t)

    @lru_cache(maxsize=None)
    def _K2(self, i, sx, tu):
        #print(i,sx,tu)
        if (i == 0):
            return 1
        if (min(len(sx), len(tu)) < i):
            return 0
        # t -> tu
        # risque
        if (sx[-1] == tu[-1]):
            s, x = sx[:-1], sx[-1]
            t, x = tu[:-1], tu[-1]
            return self.lambda_decay * (self._K2(i, sx, t) + self.lambda_decay * self._K1(i - 1, s, t))
        else:
            x = sx[-1]
            len_t = len(tu)
            stock = -1
            for j in range(len_t):
                if (tu[len_t - 1 - j] == x):
                    stock = len_t - 1 - j
                    break
            if (stock == -1):
                return self.lambda_decay ** len_t
            else:
                return self._K2(i, sx, tu[:stock + 1]) * (self.lambda_decay ** (len_t - stock - 1))

    # SECOND TECHNIC
    def ___K(self, n, s, t):
        K = []
        len_s = len(s)
        len_t = len(t)
        for i in range(n + 1):
            K_i = []
            for j in range(len_s):
                K_i_j = [0] * len_t
                K_i.append(K_i_j)
            K.append(K_i)
        for j in range(len_s):
            for k in range(len_t):
                K[0][j][k] = 1.0

        for i in range(n):
            for j in range(len_s - 1):
                cum_sum = 0.0
                for k in range(len_t - 1):
                    cum_sum = self.lambda_decay * (cum_sum + self.lambda_decay * float(s[j] == t[k]) * K[i][j][k])
                    K[i + 1][j + 1][k + 1] = self.lambda_decay * K[i + 1][j][k + 1] + cum_sum
        output = 0.0
        for i in range(n):
            for j in range(len_s):
                for k in range(len_t):
                    output += self.lambda_decay * self.lambda_decay * float(s[j] == t[k]) * K[i][j][k]
        return output

    def train(self, dataset):
        print("Training model " + self.name)
        self.text_clf = svm.SVC(kernel='precomputed')
        gram = self.gramMatrix(dataset.getTrainSet(), dataset.getTrainSet())
        self.text_clf.fit(gram, dataset.getTrainLabels())
        print("Traning done")

    def predict(self, dataset):
        gram = self.gramMatrix(dataset.getTestSet(), dataset.getTrainSet())
        Y = self.text_clf.predict(gram)
        return Y

    def ownKernel(self, s1, s2):
        return abs(len(s1) - len(s2))

    def gramMatrix(self, X, Y):
        minDim = min(len(X), len(Y))
        #print(len(X), len(Y))

        gram = np.zeros((len(X), len(Y)))
        sim_docs_kernel_value = {}
        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        # store K(s,s) values in dictionary to avoid recalculations
        for i in range(len(X)):
            #print(i)
            sim_docs_kernel_value[1][i] = self._K(self.subseq_length, X[i], X[i])

        for i in range(len(Y)):
            sim_docs_kernel_value[2][i] = self._K(self.subseq_length, Y[i], Y[i])

        for i in range(minDim):
            for j in range(minDim):
                #print(i, j)
                if gram[i][j] == 0:
                    resultKernel = self._gram_matrix_element(X[i], Y[j], sim_docs_kernel_value[1][i],
                                                             sim_docs_kernel_value[2][j])
                    gram[i][j] = resultKernel
                    gram[j][i] = resultKernel
        if (len(X) > len(Y)):
            for i in range(minDim, len(X)):
                for j in range(len(Y)):
                    #print(i, j)

                    resultKernel = self._gram_matrix_element(X[i], Y[j], sim_docs_kernel_value[1][i],
                                                             sim_docs_kernel_value[2][j])
                    gram[i][j] = resultKernel
        elif (len(Y) > len(X)):
            for i in range(len(X)):
                for j in range(minDim, len(Y)):
                    #print(i, j)

                    resultKernel = self._gram_matrix_element(X[i], Y[j], sim_docs_kernel_value[1][i],
                                                             sim_docs_kernel_value[2][j])
                    gram[i][j] = resultKernel
        return gram

    def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        if s == t:
            return 1
        else:
            try:
                return self._K(self.subseq_length, s, t) / \
                       (sdkvalue1 * sdkvalue2) ** 0.5
            except ZeroDivisionError:
                print("Maximal subsequence length is less or equal to documents' minimal length."
                      "You should decrease it")
                sys.exit(2)
