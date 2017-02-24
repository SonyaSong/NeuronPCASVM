import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn import decomposition
from sklearn import metrics

train = pd.read_csv("C:/Users/ASUS/Desktop/SVM/train.csv")

X = train.values[:, 1:]
y = train.values[:,0]

svmm = svm.SVC(kernel='rbf')

rbf_svc = (svmm.fit(X, y), y)

test = pd.read_csv("C:/Users/ASUS/Desktop/SVM/test.csv")

X_test = test.values[:, 1:]
y_test = test.values[:,0]

result = svmm.predict(X_test)

accuracy =  metrics.accuracy_score(y_test,result)

print(accuracy)

print(result)
