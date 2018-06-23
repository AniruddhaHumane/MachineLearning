import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importing Dataset
dt = pd.read_csv("Data.csv")    # Change name of Dataset CSV file
X = dt.iloc[:,:-1].values       # If last column is not the only output change this (if 3 output cols then change to -3)
Y = dt.iloc[:,-1].values        # this too

#handling missing data and NaN (Optional Do only if data is missing or NaN)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy="mean", axis=0, verbose=1)   
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Categorizing data and one hot encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEnc_X = LabelEncoder()
X[:,0] = labelEnc_X.fit_transform(X[:,0]) # this assigns 0 1 2 but this may create comparison while training machine learning modesl so we perform one hot encoding
ohe = OneHotEncoder(categorical_features = [0]) 
X = ohe.fit_transform(X).toarray()  # changing unique values of first column into multiple columns and setting values to 1 if it exists in that row
labelEnc_Y = LabelEncoder()
Y = labelEnc_Y.fit_transform(Y) #do one hot encoding if output data is not binary

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

#Feature Scaling to remove variable domination (Optional)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#should we scale dummy variables (one hot encoded) = depends on context, everything will be on same scale but we will loose which one was which and not scaling won't break your model
# we don't need to apply it to Y as it is already has only 2 values 0 and 1