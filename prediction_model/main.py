# there will be two steps out here 1- data cleaning and other will be the model training and testing
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle


def create_model(data):
     x=data.drop('diagnosis',axis=1)
     y=data['diagnosis']
     #splitting dataset into two parts
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
     #scaling the values
     scalar =StandardScaler()
     x_tr=scalar.fit_transform(x_train)
     x_te=scalar.transform(x_test)
     #model training
     classifier = LogisticRegression()
     classifier.fit(x_tr, y_train)
     #model is ready for test
     
     y_predict=classifier.predict(x_te)
     print("accuracy is :",accuracy_score(y_test,y_predict)) 
     print("classification is \n:",classification_report(y_test,y_predict))
     return  scalar,classifier


def clean_data():
      data=pd.read_csv("dataset/breast-cancer.csv")
      data =data.drop('id',axis=1)
      data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
    
      return data

def main():
      
      data=clean_data()
      scalar,model=create_model(data)
      with open('model/model.pkl','wb') as f:
           pickle.dump(model,f)

      with open('model/scalar.pkl','wb') as f:
           pickle.dump(scalar,f)
      

if __name__=='__main__':
    main()
