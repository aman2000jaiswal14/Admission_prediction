import os
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("Admission_Prediction.csv")

df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mode()[0],inplace=True)

x = df.drop(['Serial No.','Chance of Admit'],axis=1)
y = df['Chance of Admit']

from sklearn.preprocessing  import StandardScaler
std_scale = StandardScaler()
scaled_data = std_scale.fit_transform(x)

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(scaled_data,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(train_x,train_y)

print(reg.score(test_x,test_y))


scalefilename = 'stdscale.pickle'
pickle.dump(std_scale,open(scalefilename,'wb'))
filename = "adimission_chance_reg_model.pickle"
pickle.dump(reg,open(filename,'wb'))

loaded_scale = pickle.load(open(scalefilename,'rb'))
loaded_model = pickle.load(open(filename,'rb'))
a = loaded_model.predict(loaded_scale.transform([[300,100,2.0,4.0,3.0,8.2,0]]))
print(a)