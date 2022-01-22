import numpy as np
import pandas as ps
import seaborn as sbn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
dataset=ps.read_csv("heart.csv")
#print(dataset.isnull().sum()) # no null value
print(dataset.shape)
#dataset.drop_duplicates(keep=False,inplace=True) # removing duplicate in dataset

print(dataset.shape)
X=dataset.drop(columns="target",axis=1,inplace=False)
Y=dataset["target"]
#print(X.describe())
#print(X.info())
#scale=StandardScaler()
#scale.fit_transform(X,Y)
#print(X.head(10))# print first 10 row
print(dataset["target"].value_counts())


#splitting training set and test data

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)
# logistic regression
print(X.shape,Y.shape)

"""Logistic regression """
logiReg=LogisticRegression()
logiReg.fit(xtrain,ytrain)
predicted=logiReg.predict(xtrain)

"""polynomial regression """

polyReg=PolynomialFeatures(degree=3)
predictedpoly = polyReg.fit_transform(xtrain)
print("predicted Poly :",predictedpoly)
"""We cant plot ploynomial X points in model bcos polynomial regression use only single dimensional X and Y value
but here we have more x values"""



#sbn.regplot(x=Y,y=Y,data=dataset,logistic=True,ci=None)

plt.scatter(xtrain["age"],Y,color="blue")
plt.plot(xtrain,predicted,color="red")
plt.xlabel("xvalue")
plt.ylabel("ylabel")
plt.title("ply regression results")
plt.show()

#predictPoly=as.pred



accuracyScore=accuracy_score(predicted,ytrain)
print(accuracyScore)
#afterModel=logiReg.fitT
"""Buillding predictive model"""
data=[57,1,0,110,335,0,1,143,1,3,1,1,3]

inp=np.asarray(data)

inp=inp.reshape(1,-1)

print(logiReg.predict(inp))


