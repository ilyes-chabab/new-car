import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('carData.csv')
year = df['year']
selling_price = df['selling_price']

# get linear regression thanks to scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(year, selling_price)

# Afficher les résultats
print(f"Pente (slope) : {slope}")
print(f"Ordonnée à l'origine (intercept) : {intercept}")
print(f"Coefficient de corrélation (r_value) : {r_value}")
print(f"p-value : {p_value}")
print(f"Erreur standard : {std_err}")


# Tracer les points et la droite de régression
plt.scatter(year, selling_price, color='blue', label='Données')
plt.plot(year, slope * year + intercept, color='red', label='Droite de régression')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.legend()
plt.show()
