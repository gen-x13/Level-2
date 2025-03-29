""" #genxcode - LEVEL : Analyse des ventes d'une boutique en ligne """

# Importation des modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importation du dataset (source : Kaggle - "Amazon Sales Dataset")

sales = pd.read_csv('../data/amazon_sales.csv')

# Vérification et nettoyage de données
    
# | Etape 1 | : Supprimer les colonnes inutiles

sales.drop(['product_id', 'discount_percentage', 'user_id', 'user_name',
            'review_id', 'review_title', 'review_content', 'img_link', 
            'product_link', 'about_product'], axis=1, inplace=True)


print(sales.head()) #Vérifier les colonnes restantes

# | Etape 2 | : Éliminer les caractères spéciaux 
#               et modifier le type de certaines colonnes

spe_chars = r'[@#$%₹€]'  # Enlever ces caractères spéciaux

for col in ['actual_price', 'discounted_price', 'rating', 'rating_count']:
    
    if col in sales.columns:
        
        # Convertion en string et élimination des caractères spéciaux
        sales[col] = sales[col].astype(str).str.replace(spe_chars, '', regex=True)
        
        # Transformer la virgule en un point pour une conversion correcte des flottants
        sales[col] = sales[col].str.replace(',', '')
        
        # Convertion en numérique (float)
        sales[col] = pd.to_numeric(sales[col], errors='coerce')

print("Types de colonnes :", sales[['actual_price', 'discounted_price', 
                                     'rating_count', 'rating']].dtypes)

# | Étape 3 | : Vérification et nettoyage des données NaN

# Calcul du pourcentage de NaN
a = (sales.isnull().sum().sum() / sales.size) * 100  

if a == 0:
    print(f"Pas besoin de correction des Nan. {a:.2f}% NaN.")
                        # .2f : deux chiffres après le point
else:
    print(f"Il faut procéder au nettoyage des données. Il y a {a:.2f}% of NaN.")

    # Supprimer les NaN
    sales.dropna(inplace=True)

    # Vérification après le nettoyage
    a = (sales.isnull().sum().sum() / sales.size) * 100  

    if a == 0:
        print("Nettoyage fini. Il n'y a plus de données Nan")
    else:
        print(f"Avertissement : Il reste des données NaN ({a:.2f}%). Un nettoyage plus approfondi est nécessaire.")

print("Vérification et nettoyage, fait.")
print("Vérfication des Nan ", sales.isnull().sum())


#-----------------------------------------------------------------------------#

##------ Partie 1 : Trouver les produits les plus vendus 
#                   ainsi que les pics de vente sur la base du nombre de votes


"""

Caractéristiques : 
    
- product_name : Nom du produit

Target :
    
- rating_count : Nombre de personnes ayant évalué un produit

"""

# Trier les produits du plus au moins voté

sales_sorted_vote = sales.sort_values(by='rating_count', ascending=False)


# Création de la figure

plt.figure(figsize=(12, 6))

# Raccourcir le nom des produits

sales_sorted_vote['product_name'] = sales_sorted_vote['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:50] + '...')

# Différencier les votes les plus nombreux des moins nombreux

c = ['black' if i > 3 else 'purple' for i in range(50)]

# Trier les noms et le nombre d'évaluations (graphique en forme de sucette)
  
plt.hlines(
    sales_sorted_vote.iloc[0:50]['product_name'],
    sales_sorted_vote.iloc[0:50]['rating_count'],
    xmax=0,
    color=c,
    alpha=0.5
)

# Titres et appellations

plt.ylabel("Noms des produits")
plt.xlabel("Nombres de votes")
plt.title("Produit le plus vendu, sur la base du nombre d'évaluations")


plt.tight_layout() # Ajustement de l'affichage

plt.show() # Afficher le graphique

# Montrer les produits les plus vendus

print('Les trois produits les plus vendus sont :', 
      sales_sorted_vote.iloc[:3]['product_name'])

# Observation et interprétation

print("Observation : sur la base du nombre de votes,")  
print("Nous pouvons constater que les câbles d'ordinateur et de télévision ont un nombre très élevé de votes.")
print("Interprétation : Les clients semblent préférer les câbles HDMI")
print("et d'autres produits technologiques, qu'ils peuvent se procurer à bas prix sur Amazon.")

#-----------------------------------------------------------------------------#

##------ Partie 2 : Observer si les produits plus chers se vendent moins

"""

Caractéristiques : 
    
- rating_count

Cible :
    
- prix réel

"""


# Trier les prix des produits du plus cher au moins cher

sales_sorted_price = sales.sort_values(by='actual_price', ascending=False)


# Création de deux graphiques

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Sélection des produits chers et moins chers

expensive = sales_sorted_price.iloc[:20]
cheap = sales_sorted_price.iloc[-20:]


expensive['product_name'] = expensive['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:15] + '...') 

cheap['product_name'] = cheap['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:15] + '...') 

# Colormap avec Seaborn

c0 = sns.cubehelix_palette(as_cmap=True)
c1 = sns.cubehelix_palette(as_cmap=True)

colors = np.random.rand(20) # Valeur pour la colormap


# Graphique des produits chers

axes[0].hlines(y=expensive['product_name'], xmin=0, xmax=expensive['rating_count'], color='grey')
axes[0].scatter(expensive['rating_count'], expensive['product_name'], c= colors, cmap=c0, marker="^")  # Ajout des points

axes[0].set_xlabel("Nombre de votes")
axes[0].set_title("Produits chers")

# Graphique des produits moins chers

axes[1].hlines(y=cheap['product_name'], xmin=0, xmax=cheap['rating_count'], color='grey')
axes[1].scatter(cheap['rating_count'], cheap['product_name'], c= colors, cmap=c1, marker="v")  # Ajout des points

axes[1].set_xlabel("Nombre de votes")
axes[1].set_title("Produits moins chers")

plt.title("Comparer la vente de produits chers et bon marché")

plt.tight_layout()
plt.show()


# Observation and interprétation

print("Observation : sur la base du nombre de votes,")  
print("On constate que seuls deux produits onéreux ont une note supérieure à celle de")
print("les produits bon marché. Mais la plupart des produits chers ont une note plus basse",
      "que les produits bon marché.")
print("Interprétation : Les clients semblent préférer les produits moins chers aux produits plus onéreux,")
print("sauf lorsqu'il s'agit de smartphones et de téléviseurs.")


#-----------------------------------------------------------------------------#

##------ Partie 3 : Trouver la catégorie qui se vend le mieux et qui obtient le plus de votes


"""

Caractéristiques : 
    
- category

Cible :
    
- rating count

"""

# Total des votes pour chaque catégorie

total = sales.groupby("category")['rating_count'].sum()


# Tri de la série Total

total_sorted_categorymost = total.sort_values(ascending=False) 

# Transforme "category" en un Dataset

total_sorted_categorymost = total_sorted_categorymost.reset_index()


total_sorted_categorymost['category'] = total_sorted_categorymost['category'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:30] + '...')



plt.figure(figsize=(12, 6))

c = ['blue' if i > 1 else 'red' for i in range(25)]

plt.hlines(
    total_sorted_categorymost['category'],
    total_sorted_categorymost['rating_count'],
    xmax=0,
    color=c,
    alpha=0.5
    )

plt.title("Catégories les plus vendues et les plus votées")
plt.xlabel("Nombre de votes")
plt.ylabel("Catégories")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# Observation and interprétation

print("Observation : sur la base du nombre de votes,")  
print("Nous pouvons constater que la catégorie électronique est la plus choisie par les clients.")
print("En particulier, tout ce qui concerne", total_sorted_categorymost.iloc[:3]["category"])
print("Alors que la catégorie 'Car' est la moins choisie, avec seulement : ", 
      total_sorted_categorymost.iloc[-1]['rating_count'], "votes")
print("Interprétation : Cela confirme que les clients ont une préférence pour l'électronique. ")
print("produits, en particulier : écouteurs/écouteurs, accessoires pour ordinateurs, accessoires", 
      "pour téléphones portables et accessoires pour téléviseurs.")


#-----------------------------------------------------------------------------#


##------ Partie 4 : Conclusion avec une visualisation sur Seaborn


sales['category'] = sales['category'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:20] + '...')


sns.catplot(x=sales['rating_count'].iloc[:25:2].sort_values(ascending=False), 
                y=sales['actual_price'],data=sales, hue='category')

sns.catplot(x=sales['rating_count'].iloc[:25:2].sort_values(ascending=False), 
                y=sales['rating'],data=sales, hue='category')


print("Observation finale : En fonction du nombre de votes et du prix,")  
print("On constate que plus un produit est bon marché, plus il a de chances d'être choisi par les clients.")
print("En particulier, les catégories d'électronique avec un vote d'évaluation élevé et un prix bas.")
print("Cependant, nous pouvons observer que la même catégorie a un taux élevé et le plus grand nombre de votes.")
print("Conclusion : cela confirme que les clients ont une préférence pour la catégorie électronique sur Amazon. ")
print("Spécifiquement : écouteurs/écouteurs, accessoires pour ordinateurs, accessoires pour téléphones portables et accessoires pour téléviseurs,")
print("Les consommateurs ont besoin de tous les accessoires qu'ils peuvent se procurer à bas prix en ligne et être satisfaits de la qualité du produit.")
















































