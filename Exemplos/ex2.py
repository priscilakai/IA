# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 08:54:43 2021

@author: pri_k
"""

from keras.models import Sequential
from keras import layers
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

iris = load_iris()

X_data = iris.data #4 características
y_data = iris.target #3 classes

#construindo o modelo
model = Sequential([
            layers.Dense(5, kernel_regularizer=l1(0.01), activation="relu",input_shape=(4,)),
            layers.Dense(5, kernel_regularizer=l1(0.01), activation="relu"),
            layers.Dropout((0.1)),
            layers.Dense(3, activation="softmax")  
            ])

(X, Xtest,y,ytest) = train_test_split(X_data,y_data, test_size=0.2, random_state=42)

model.compile(optimizer='SGD')
model.fit(X, y, epochs=5, batch_size=5)
predictions = model.predict(Xtest)

print(classification_report(ytest, predictions.argmax(axis=1),labels=[0,1,2]))