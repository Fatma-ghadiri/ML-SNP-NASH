import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

##    This script implements knn on the SNP dataset.
##    Input:
##		file: str
##			Encoded dataset
##	

data=pd.read_csv(r"C:\Normalize_data.csv")
y=data.Hedef
x=data[['rs738409','rs3813867','rs1260326','Gender']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Chi squre & KNN Algoritma')
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
