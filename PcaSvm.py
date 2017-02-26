import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn import decomposition
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

#PCA
mark = pd.read_csv("C:/Users/ASUS/Desktop/1.csv")

mark_X = mark.values[:,1:]
mark_y = mark.values[:,0]
mark2_X = mark.values[:,1:]
mark3_X = mark.values[:,1:]

pca = decomposition.PCA(n_components=10)
pca.fit(mark_X)
mark_X = pca.transform(mark_X)

#SVM
X_train, X_test, y_train, y_test = cross_validation.train_test_split(mark_X, mark_y, test_size=0.33, random_state=0)

#kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
#1.‘linear’: 0.97
#2.‘rbf’: 0.98
#3.‘poly’: 1.0
#4.‘sigmoid’: 0.27
#5.‘precomputed’: error
svmm = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,

tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)

rbf_svc = (svmm.fit(X_train, y_train), y_train)

result = svmm.predict(X_test)

accuracy =  metrics.accuracy_score(y_test,result)

print("accuracy: ")
print(accuracy)

#print(result)



# plt
pca2 = decomposition.PCA(n_components=2)
pca2.fit(mark2_X)
mark2_X = pca.transform(mark2_X)

x_min, x_max = mark2_X[:, 0].min() - .5, mark2_X[:, 0].max() + .5
y_min, y_max = mark2_X[:, 1].min() - .5, mark2_X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(mark2_X[:, 0], mark2_X[:, 1], c=mark_y, cmap=plt.cm.Paired)
plt.xlabel('Features 1')
plt.ylabel('Features 2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(mark3_X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=mark_y,
           cmap=plt.cm.Paired)
#ax.set_title("First three PCA directions")
ax.set_xlabel("Features 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Features 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Features 3")
ax.w_zaxis.set_ticklabels([])

plt.show()