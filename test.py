import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 


#data loading
df = pd.read_csv('data/carData.csv')
Year = df['Year'].values.reshape(-1,1)
Selling_Price = df['Selling_Price'] * 1000
Transmission = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
Kms_Driven = df['Kms_Driven']
Present_Price = df['Present_Price'] * 1000
Fuel_Type = df['Fuel_Type'].map({'Petrol' :0, 'Diesel' :1, 'CNG' : 2})
Seller_Type = df['Seller_Type'].map({'Dealer':0 , 'Individual' : 1})
Owner = df['Owner']

#split dataset on 80% data for train and 20% data for test 

X_train, X_test, y_train, y_test = train_test_split(np.column_stack((Year, Transmission, Kms_Driven, Present_Price, Fuel_Type, Seller_Type)), Selling_Price, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#model creation
model = LinearRegression()

model.fit(X_train,y_train)

score = model.score(X_test,y_test)
print('score '+str(score))

prediction = model.predict(X_test)
print(f' prediction {prediction} ' )