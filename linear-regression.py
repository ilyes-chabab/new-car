import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 

#data loading
df = pd.read_csv('data/carData.csv')
year = df['Year'].values.reshape(-1,1)
selling_price = df['Selling_Price'].values * 1000

#split dataset on 80% data for train and 20% data for test 
X_train, X_test, y_train, y_test = train_test_split(year, selling_price, test_size=0.2, random_state=0)
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

#we sort the datas to have a right line instead of several peaks
sorted_indices = np.argsort(X_test.flatten())  
X_test_sorted = X_test[sorted_indices]
prediction_sorted = prediction[sorted_indices]

#display with my plot lib

plt.scatter(X_test, y_test, label='Données réelles', color='b')  # Points réels
plt.plot(X_test_sorted, prediction_sorted, color='r', label='Régression linéaire')  # Droite de régression
plt.xlabel('Année')
plt.ylabel('Prix de vente')
plt.legend()
plt.show()
