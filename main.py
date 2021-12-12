import json
import os
import numpy as np
from scipy.io import loadmat
import re
import sklearn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ck_train_path = "data/CK/CKTrain"
ck_test_path = "data/CK/CKTest"

nasa_train_path = "data/NASA/NASATrain"
nasa_test_path = "data/NASA/NASATest"

result_path="result.txt"
with open(result_path,'w') as file:
    file.write("result"+"\n")
    file.close()


# mlp+decisiontree

# load data
def load(path, name):
    data = loadmat(path)[name]
    return data[:, 0][0], data[:, 1][0], data[:, 2][0]


def mlp(train_data, test_data):
    ans = {}
    n, m = train_data.shape
    # str to int
    x_train, y_train = train_data[:, 0:m - 1], (train_data[:, m - 1] > 0).astype(int)
    x_test, y_true = test_data[:, 0:m - 1], (test_data[:, m - 1] > 0).astype(int)

    clf = MLPClassifier(random_state=0,max_iter=5000)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    try:
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        for i, label in enumerate(y_true):
            if label:
                if prediction[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction[i]:
                    FP += 1
                else:
                    TN += 1
        if (FP + TN) == 0:
            pf = "no FP and TN."
        else:
            pf = FP / (FP + TN)
        try:
            auc = roc_auc_score(y_true, prediction)
        except ValueError as e:
            auc = str(e)
        ans['method'] = 'mlp'
        ans['pf'] = pf
        ans['auc'] = auc
        ans['recall'] = recall_score(y_true, prediction)
        ans['accuracy'] = accuracy_score(y_true, prediction)
        ans['F-measure'] = f1_score(y_true, prediction)
        ans['precision'] = precision_score(y_true, prediction)

        with open(result_path, 'a') as file:
            file.write(str(ans)+"\n")
            file.close()
    except ValueError as e:
        print(str(e))


def decisiontree(train_data, test_data):
    ans = {}
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
        prediction = clf.predict(x_test)
        for i, label in enumerate(y_true):
            if label:
                if prediction[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction[i]:
                    FP += 1
                else:
                    TN += 1
        if (FP + TN) == 0:
            pf = "no FP and TN."
        else:
            pf = FP / (FP + TN)
        try:
            auc = roc_auc_score(y_true, prediction)
        except ValueError as e:
            auc = str(e)
        ans['method'] = 'DecisionTree'
        ans['pf'] = pf
        ans['auc'] = auc
        ans['recall'] = recall_score(y_true, prediction)
        ans['accuracy'] = accuracy_score(y_true, prediction)
        ans['F-measure'] = f1_score(y_true, prediction)
        ans['precision'] = precision_score(y_true, prediction)

        with open(result_path, 'a') as file:
            file.write(str(ans)+"\n")
            file.close()
    except ValueError as e:
        print(str(e))


if __name__ == '__main__':
    CK_Train = os.listdir(ck_train_path)
    CK_Test = os.listdir(ck_test_path)
    NASA_Train = os.listdir(nasa_train_path)
    NASA_Test = os.listdir(nasa_test_path)
    print(CK_Test, CK_Train, NASA_Train, NASA_Test)
    # load data
    # ck
    with open(result_path, 'a') as file:
        file.write("ck" + "\n")
        file.close()
    for index in range(len(CK_Test)):
        train_data1, train_data2, train_data3 = load(os.path.join(ck_train_path, CK_Train[index]),
                                                     re.match(r'\w*', CK_Train[index]).group())
        test_data1, test_data2, test_data3 = load(os.path.join(ck_test_path, CK_Test[index]),
                                                  re.match(r'\w*', CK_Test[index]).group())
        print(re.match(r'\D*\d', CK_Train[index]).group())
        with open(result_path, 'a') as file:
            file.write(re.match(r'\D*\d', CK_Train[index]).group()+"\n")
            file.close()
        mlp(train_data1, test_data1)
        decisiontree(train_data1, test_data1)

        mlp(train_data2, test_data2)
        decisiontree(train_data2, test_data2)

        mlp(train_data3, test_data3)
        decisiontree(train_data3, test_data3)

    # nasa
    with open(result_path, 'a') as file:
        file.write("nasa" + "\n")
        file.close()
    for index in range(len(NASA_Test)):
        train_data1, train_data2, train_data3 = load(os.path.join(nasa_train_path, NASA_Train[index]),
                                                     re.match(r'\w*', NASA_Train[index]).group())
        test_data1, test_data2, test_data3 = load(os.path.join(nasa_test_path, NASA_Test[index]),
                                                  re.match(r'\w*', NASA_Test[index]).group())
        print(re.match(r'\D*\d', NASA_Train[index]).group())
        with open(result_path, 'a') as file:
            file.write(re.match(r'\D*\d', NASA_Train[index]).group()+"\n")
            file.close()
        mlp(train_data1, test_data1)
        decisiontree(train_data1, test_data1)

        mlp(train_data2, test_data2)
        decisiontree(train_data2, test_data2)

        mlp(train_data3, test_data3)
        decisiontree(train_data3, test_data3)
