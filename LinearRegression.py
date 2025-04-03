import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

"""
    This class make a linear regression with your feature(s) and your target

    Methods:
        split_train_and_test(x,y,test_size,random_seed):
        split your dataset (x,y) between train and test part ,
        test_size is the rate of split (test_size = 0.2 is 20% test and 80% train) 
        random_seed is a number who will "freeze" the random . with en random_seed=1 you will have the same result  
        to each time , if random_seed is None , so the result will be random at each time


        train_model(x_train,y_train): 
        this compute theta_0 and theta_1. this variables are the prediction value and the variation rate
        , it will be useful to have the prediction and score

        prediction(x_test):
        this will predict the y value with x 

        score(x_test,y_test) :
        this will return the least square , it's the score

        display_graph(x_test,y_test):

        this will display the graph if it's a univariate linear regression,  thanks to myplotlib 


"""

class LinearRegression:    
    
    def split_train_and_test(self,x,y,test_size,random_seed):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        indices = np.random.permutation(len(x))

        test_indice_size = int(len(x) * test_size)

        test_indices = indices[:test_indice_size]
        train_indices = indices[test_indice_size:]

        x_train ,y_train, x_test ,y_test = x[test_indices] , y[test_indices] , x[train_indices] , y[train_indices]
        return x_train ,y_train, x_test ,y_test

    def train_model(self,x_train,y_train):
        n = len(x_train)

        if len(x_train) != len(y_train):
            raise ValueError("Les listes x et y doivent avoir la même longueur.")
        
        # the mean
        mean_x = sum(x_train) / n
        mean_y = sum(y_train) / n

        # calcul of theta_0 and theta_1
        num = sum((x_train.iloc[i] - mean_x) * (y_train.iloc[i] - mean_y) for i in range(n))
        den = sum((x_train.iloc[i] - mean_x) ** 2 for i in range(n))
        self.theta_1 = num / den
        self.theta_0 = mean_y - self.theta_1 * mean_x

    def prediction(self,x_test):
        return self.theta_0 + self.theta_1 * x_test

    def score(self,x_test,y_test):
        mean_y = sum(y_test) / len(y_test)
        ss_tot = sum((yi - mean_y) ** 2 for yi in y_test)
        ss_res = sum((y_test.iloc[i] - self.prediction(x_test.iloc[i])) ** 2 for i in range(len(x_test)))
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def display_graph(self,x_test,y_test):
        plt.scatter(x_test, y_test, label='Données', color='blue')
        y_pred = [self.prediction(xi) for xi in x_test]
        print(f"La valeur de prédiction est de {self.theta_0} et le taux de variation est de {self.theta_1}")
        plt.plot(x_test, y_pred, color='red', label='Droite de régression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Régression linéaire')
        plt.legend()
        plt.show()
    
df = pd.read_csv('data/carData.csv')
Year = df['Year']
Selling_Price = df['Selling_Price'] * 1000
Transmission = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
Kms_Driven = df['Kms_Driven']
Present_Price = df['Present_Price'] * 1000
Fuel_Type = df['Fuel_Type'].map({'Petrol' :0, 'Diesel' :1, 'CNG' : 2})
Seller_Type = df['Seller_Type'].map({'Dealer':0 , 'Individual' : 1})
Owner = df['Owner']

model= LinearRegression()
x_train ,y_train, x_test ,y_test = model.split_train_and_test(Year,Selling_Price,test_size=0.2,random_seed=1)

model.train_model(x_train ,y_train)
print(model.score(x_test,y_test))
model.display_graph(x_test ,y_test)

