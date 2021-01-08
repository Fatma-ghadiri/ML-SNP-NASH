import numpy as np 
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
#Recursive Feature Elimination- SVM
data=pd.read_csv(r"C:\Normalize_data.csv")

features = [f for f in data.columns if f not in ['Hedef','SK_ID_CURR']]
X = data[features].values
y= data['Hedef'].values.ravel()
svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 5)
rfe = rfe.fit(X, y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
