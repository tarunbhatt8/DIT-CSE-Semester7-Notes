#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

#Missing Data
"""from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])"""

#Categorical Data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""

#Splitting dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=1/3,random_state=0)

#Scaling
"""from sklearn.preprocessing import StandardScalar
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#fitting linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred=regressor.predict(X_test)
#Visualization(train)
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('salary vs experience(train)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#Visualization(test)
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('salary vs experience(test)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()