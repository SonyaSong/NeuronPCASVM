import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn import decomposition

mark = pd.read_csv("C:/Users/ASUS/Desktop/3f/4c.csv")
marks=mark.values[:,1:]
pca = decomposition.PCA(n_components=10)
pca.fit(marks)
marks = pca.transform(marks)



np.savetxt("C:/Users/ASUS/Desktop/pcaDone/4c.csv",marks,delimiter=",")

print(marks)
