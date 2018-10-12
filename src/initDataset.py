from html.parser import HTMLParser
import os.path
from nltk.corpus import stopwords
import string

trainSetName = 'train.txt'
testSetName = 'test.txt'


class MyHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.lastTag = ''
        self.topics = True
        self.topicsList = []
        self.text = ""
        self.trainSet = []
        self.testSet = []
        self.lewisSplit = ""
        self.noTopic = 0

    def handle_starttag(self, tag, attrs):
        if tag != "d":
            self.lastTag = tag
        if tag == "reuters":
            self.topicsList = []
            for attr in attrs:
                if attr[0] == "topics":
                    if attr[1] == "YES":
                        self.topics = True
                    else:
                        self.topics = False
                elif attr[0] == "lewissplit":
                    self.lewisSplit = attr[1]

    def handle_endtag(self, tag):
        if tag == "reuters" and self.topics:
            if self.lewisSplit == "TEST":
                self.testSet.append([self.text, self.topicsList])
            if self.lewisSplit == "TRAIN":
                if len(self.topicsList) == 0:
                    self.noTopic += 1
                self.trainSet.append([self.text, self.topicsList])

    def handle_data(self, data):
        if self.lastTag == "body" and self.topics:
            data = (' '.join(data.split())).strip()
            if data != "":
                self.text = data
        if self.lastTag == "topics" and self.topics:
            data = data.strip()
            if data != "":
                self.topicsList.append(data)

    def getSets(self):
        return self.trainSet, self.testSet

def removeStopWords(s):
    s = s.lower()
    s = ' '.join([word for word in s.split() if word not in (stopwords.words('english')+["reuter"])])
    for p in string.punctuation:
        s = s.replace(p,"")
    return s

def extractDataSet():
    # check if the file existing
    # if yes use files
    # if no do this
    trainSet = []
    testSet = []
    if(os.path.isfile(trainSetName) and os.path.isfile(testSetName)):
        with open(trainSetName) as f:
            lines = f.readlines()
            for body, cat in zip(lines[0::2], lines[1::2]):
                trainSet.append(
                    [body.replace('\n', ''), cat.replace('\n', '').split(";")])
        f.close()
        with open(testSetName) as f:
            lines = f.readlines()
            for body, cat in zip(lines[0::2], lines[1::2]):
                testSet.append(
                    [body.replace('\n', ''), cat.replace('\n', '').split(";")])
        f.close()
        return trainSet, testSet
    else:
        parser = MyHTMLParser()

        for i in range(22):
            print("Treatment of document " + str(i))
            if i < 10:
                with open('../reuters21578/reut2-00' + str(i) + '.sgm', encoding="ISO-8859-1") as f:
                    read_data = f.read()
                f.close()
            else:
                with open('../reuters21578/reut2-0' + str(i) + '.sgm', encoding="ISO-8859-1") as f:
                    read_data = f.read()
                f.close()

            parser.feed(read_data)

        trainSet, testSet = parser.getSets()
        print("Saving train set")
        with open(trainSetName, 'w') as f:
            for i in range(len(trainSet)):
                f.write(removeStopWords(trainSet[i][0]) + '\n')
                f.write(';'.join(trainSet[i][1]) + '\n')
        f.close()
        print("Saving test set")
        with open(testSetName, 'w') as f:
            for i in range(len(testSet)):
                f.write(removeStopWords(testSet[i][0]) + '\n')
                f.write(';'.join(testSet[i][1]) + '\n')
        f.close()

        return trainSet, testSet
