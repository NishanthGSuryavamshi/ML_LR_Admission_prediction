
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split


import seaborn as sns

sns.set()
data =pd.read_csv(r'C:\Users\vamsh\Downloads\LinearRegression-master (1)\LinearRegression-master\LinearRegressionTillCloud\Admission_Prediction.csv')

data['GRE Score']=data['GRE Score'].fillna(data['GRE Score'].mode()[0])
data['TOEFL Score']=data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['University Rating']=data['University Rating'].fillna(data['University Rating'].mean())
data= data.drop(columns = ['Serial No.'])

y=data['Chance of Admit']
x=data.drop(columns=['Chance of Admit'])

#scaler=StandardScaler()
#x_scaled=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=355)
lr=LinearRegression()
lr.fit(x_train,y_train)
import pickle
filename='final_regression_model_3.pickle'
pickle.dump(lr,open(filename,'wb'))
loaded_model=pickle.load(open(filename,'rb'))
a=loaded_model.predict([[290,110,5,5,5,10,1]])
print(a)