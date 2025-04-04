import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

class LinearRegression:    
    
    def split_train_and_test(self, x, y, test_size, random_seed):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Mélanger les indices pour obtenir des ensembles aléatoires
        indices = np.random.permutation(len(x))
        test_size_index = int(len(x) * test_size)
        
        test_indices = indices[:test_size_index]
        train_indices = indices[test_size_index:]
        
        x_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
        x_test, y_test = x.iloc[test_indices], y.iloc[test_indices]
        
        return x_train, y_train, x_test, y_test

    def train_model(self, x_train, y_train):
        # Ajouter un terme de biais (interception) en ajoutant une colonne de 1s à x_train
        x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]  # Ajouter une colonne de 1
        self.theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)

    def prediction(self, x_test):
        # Ajouter un terme de biais (interception) à x_test
        x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]  # Ajouter une colonne de 1
        return x_test.dot(self.theta)

    def score(self, x_test, y_test):
        y_pred = self.prediction(x_test)
        mean_y = np.mean(y_test)
        ss_tot = np.sum((y_test - mean_y) ** 2)
        ss_res = np.sum((y_test - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def display_graph(self, x_test, y_test):
        plt.scatter(x_test, y_test, label='Données', color='blue')
        y_pred = self.prediction(x_test)
        plt.plot(x_test, y_pred, color='red', label='Droite de régression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Régression linéaire')
        plt.legend()
        plt.show()


# Charger les données
df = pd.read_csv('data/carData.csv')
Year = df['Year'].values
Present_Price = df['Present_Price'].values * 1000
Selling_Price = df['Selling_Price'].values * 1000

# Créer la matrice X (2D) avec Year et Present_Price
X = pd.DataFrame({'Year': Year, 'Present_Price': Present_Price})

# Initialiser le modèle
model = LinearRegression()

# Diviser les données en ensembles d'entraînement et de test
x_train, y_train, x_test, y_test = model.split_train_and_test(X, pd.Series(Selling_Price), test_size=0.2, random_seed=1)

# Entraîner le modèle
model.train_model(x_train, y_train)

# Calculer et afficher le score
print("Score R²:", model.score(x_test, y_test))



