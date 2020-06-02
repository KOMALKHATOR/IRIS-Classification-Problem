# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:28:39 2020

@author: Komal
"""
from sklearn import datasets

iris= datasets.load_iris()
X=iris.data
Y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


#Train the svc

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


#rbf with gamma=1
svc=SVC(kernel='rbf',gamma=1.0)
svc.fit(X_train,Y_train)
Y_predict = svc.predict(X_test) 
cm_rbf01 = confusion_matrix(Y_test,Y_predict)


#it's best amongst all so we will use score here
score=svc.score(X_test,Y_test)



#rbf with gamma =10
svc=SVC(kernel='rbf',gamma=10.0)
svc.fit(X_train,Y_train)
Y_predict = svc.predict(X_test)
cm_rbf10 = confusion_matrix(Y_test,Y_predict)

#kernel=linear
svc=SVC(kernel='linear')
svc.fit(X_train,Y_train)
Y_predict=svc.predict(X_test)
cm_linear=confusion_matrix(Y_test,Y_predict)

#kernel=poly
svc=SVC(kernel='poly')
svc.fit(X_train,Y_train)
Y_predict=svc.predict(X_test)
cm_poly=confusion_matrix(Y_test,Y_predict)


#kernal=sigmoid
svc=SVC(kernel='sigmoid')
svc.fit(X_train,Y_train)
Y_predict=svc.predict(X_test)
cm_sigmoid=confusion_matrix(Y_test,Y_predict)
