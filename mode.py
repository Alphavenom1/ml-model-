# ml-model-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import warnings
sns.set()
warnings.filterwarnings('ignore')
company=pd.read_csv('company_after_enigeering.csv')
company.info
company.status.value_counts()
X=company.drop('status',axis=1)
y=company.status
company.status
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
classifier=LogisticRegression(solver='liblinear',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
classifier.score(X_test,y_test)
X2=company.drop('isClosed',axis=1)
y2=company.isClosed
company.isClosed
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y,test_size=0.2,random_state=0)
classifier2=LogisticRegression(solver='liblinear',random_state=0)
classifier2.fit(X2,y2)
y2_pred=classifier.predict(X2_test)
classifier.score(X2_test,y2_test)
def model_pred(model, name='Default'):
    model.fit(X2_train,y2_train)
    preds=model.predict(X2_test)
    print('_',name,'_','\n',confusion_matrix(y2_test,preds),'\n','Accuracy:', round(accuracy_score(y2_test,preds), 5),'\n')
lr=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_pred(lr,'LogisticRegression')
