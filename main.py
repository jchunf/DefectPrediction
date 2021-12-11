import json
import os
import numpy as np
from scipy.io import loadmat
import re
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ck_train_path = "data/CK/CKTrain"
ck_test_path = "data/CK/CKTest"

nasa_train_path = "data/NASA/NASATrain"
nasa_test_path = "data/NASA/NASATrain"


# mlp+decisiontree

# load data
def load(path, name):
    data = loadmat(path)[name]
    print(name)
    return data[:, 0][0], data[:, 1][0], data[:, 2][0]


def mlp(train_data, test_data):
    n,m=train_data.shape
    #str to int
    x_train,y_train = train_data[:,0:m-1],(train_data[:,m-1]>0).astype(int)
    x_test,y_true= test_data[:,0:m-1],(test_data[:,m-1]>0).astype(int)

    clf=MLPClassifier(random_state=5,max_iter=5000)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    try:
        clf.fit(x_train,y_train)
        predicton= clf.predict(x_test)
        for i, label in enumerate(y_true):
            if label:
                if predicton[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if predicton[i]:
                    FP += 1
                else:
                    TN += 1
        if (FP + TN) == 0:
            pf = "no negative samples."
        else:
            pf = FP / (FP + TN)
        try:
            auc = roc_auc_score(y_true, predicton)
        except ValueError as e:
            auc = str(e)
    except ValueError as e:
        print(str(e))
    print(pf,auc)


def decisiontree(train_data, test_data):
    n, m = train_data.shape
    # str to int
    x_train, y_train = train_data[:, 0:m - 1], (train_data[:, m - 1] > 0).astype(int)
    x_test, y_true = test_data[:, 0:m - 1], (test_data[:, m - 1] > 0).astype(int)

    clf = DecisionTreeClassifier(random_state=5)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    try:
        clf.fit(x_train, y_train)
        predicton = clf.predict(x_test)
        for i, label in enumerate(y_true):
            if label:
                if predicton[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if predicton[i]:
                    FP += 1
                else:
                    TN += 1
        if (FP + TN) == 0:
            pf = "no negative samples."
        else:
            pf = FP / (FP + TN)
        try:
            auc = roc_auc_score(y_true, predicton)
        except ValueError as e:
            auc = str(e)
    except ValueError as e:
        print(str(e))
    print(pf, auc)


if __name__ == '__main__':
    CK_Train = os.listdir(ck_train_path)
    CK_Test = os.listdir(ck_test_path)
    NASA_Train = os.listdir(nasa_train_path)
    NASA_Test = os.listdir(nasa_test_path)
    print(CK_Test, CK_Train, NASA_Train, NASA_Test)
    # load data
    for index in range(len(CK_Test)):
        train_data1, train_data2, train_data3 = load(os.path.join(ck_train_path, CK_Train[index]),
                                                     re.match(r'\w*', CK_Train[index]).group())
        test_data1, test_data2, test_data3 = load(os.path.join(ck_test_path, CK_Test[index]),
                                                  re.match(r'\w*', CK_Test[index]).group())
        mlp(train_data1,test_data1)
        decisiontree(train_data1,test_data1)

        mlp(train_data2,test_data2)
        decisiontree(train_data1,test_data1)

        mlp(train_data3,test_data3)
        decisiontree(train_data1,test_data1)