# Contexte :

Nous avons pour but d'estimer le prix d'une voiture d'occasion selon plusieurs critère que nous avons dans le dataset ci-dessous :
![dataset](pictures\dataset_picture.png)

# requis :

Pour se faire nous nous sommes servis des librairies : 

Pandas : afin de traiter les données du dataset .

numpy : afin d'avoir d'effectuer nos calculs mathématique efficacement et de pouvoir gérer des matrices .

scikit-learn : afin de créer le model , l'entrainer et d'avoir l'estimation du prix de la voiture.



# Exploration des fichiers :


○ **Car_name** (String) : définit le nom de la voiture.

○ **Year** (int): définit l’année de fabrication de la voiture.

○ **Selling_Price** (int): définit le prix auquel le propriétaire souhaite
vendre la voiture (target). **Il faut multiplier par 1000 pour avoir le bon prix**

○ **Present_Price** (int): définit le prix de la voiture départ-usine de la
voiture. **Il faut multiplier par 1000 pour avoir le bon prix**

○ **Kms_Driver** (int): définit la distance parcourue en km par la voiture.

○ **Fuel_Type** (String): définit le type de carburant de la voiture.

○ **Seller_type** (String): définit si le vendeur est un revendeur ou un
particulier.

○ **Transmission** (String): définit si la boîte à vitesse de la voiture est
manuelle ou automatique.

○ **Owner** (int): définit le nombre d'anciens propriétaires de la voiture.


# Bonus :

En bonus , nous devions créer notre propres class de regression linéaire , pour se faire nous avons utiliser la formule des moindres carrés . 


Où theta 1 = 

![ theta 1](pictures\theta_1_formula.png)

Et theta 0 = 

![ theta 0](pictures\theta_0_formula.png)





