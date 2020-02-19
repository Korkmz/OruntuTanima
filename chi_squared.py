import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

file_data = pd.read_csv('data/odev.csv')
group_ids=file_data.iloc[:,0]
raw_data=file_data.iloc[:,1:-1]
result_data=file_data.iloc[:,-1]

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(raw_data)
chi_vals, p_vals = chi2(X,result_data)

n=10
max_val = np.argsort(chi_vals)[::-1][:n]


