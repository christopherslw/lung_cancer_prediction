import pandas as pd    #   Data processing
import numpy as np     #   Linear algebra
from sklearn.svm import SVC     #   SVM algorithm
from sklearn.metrics import accuracy_score    #   Test results
from sklearn.preprocessing import StandardScaler    #   Scales data
from sklearn.model_selection import train_test_split    #   Splits data set into training and testing
from sklearn.neighbors import KNeighborsClassifier   #   KNN algorithm

df = pd.read_csv("/Users/christopherwagner/Downloads/survey lung cancer.csv")
print(df.head())
print(df.isnull().sum())  #  Our data has no null values

df['LUNG_CANCER'].replace(['YES', 'NO'],[1, 0], inplace=True)   #   Changing categorical data to binary
df['GENDER'].replace(['F', 'M'],[1, 0], inplace=True)
print(df.head())

X = df.drop(['LUNG_CANCER'], axis = 1)   #   Keep all except for target values
Y = df['LUNG_CANCER']   #   Target values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
cols = x_train.columns
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = pd.DataFrame(x_train, columns=[cols])

svc=SVC(kernel='rbf', C=1.0)
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print('Model accuracy score with radial basis function kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('KNN model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
