""" #genxcode - LEVEL : Analysis of online store sales """

# Importing modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset (source : Kaggle - "Amazon Sales Dataset")

sales = pd.read_csv('../data/amazon_sales.csv')

# Data verification and cleansing
    
# | Step 1 | : Supressing useless columns

sales.drop(['product_id', 'discount_percentage', 'user_id', 'user_name',
            'review_id', 'review_title', 'review_content', 'img_link', 
            'product_link', 'about_product'], axis=1, inplace=True)


print(sales.head()) #Checking the columns in the dataset.

# | Step 2 | : Eliminate special characters and change the type of certain columns

spe_chars = r'[@#$%₹€]'  # Remove these special characters

for col in ['actual_price', 'discounted_price', 'rating', 'rating_count']:
    
    if col in sales.columns:
        
        # Convert to string and remove special characters
        sales[col] = sales[col].astype(str).str.replace(spe_chars, '', regex=True)
        
        # Transform the comma into a point for proper float conversion
        sales[col] = sales[col].str.replace(',', '')
        
        # Convert to numeric (float)
        sales[col] = pd.to_numeric(sales[col], errors='coerce')

print("Types of columns :", sales[['actual_price', 'discounted_price', 
                                     'rating_count', 'rating']].dtypes)

# | Step 3 | : Verification and cleaning of NaN data

# Calculation of NaN percentage
a = (sales.isnull().sum().sum() / sales.size) * 100  

if a == 0:
    print(f"No need for a NaN correction. {a:.2f}% NaN.")
                        # .2f : two digits after the decimal point
else:
    print(f"Need to proceed with data cleaning. There's {a:.2f}% of NaN.")

    # Remove NaN
    sales.dropna(inplace=True)

    # Checking after cleaning
    a = (sales.isnull().sum().sum() / sales.size) * 100  

    if a == 0:
        print("Cleaning done. No more NaN data.")
    else:
        print(f"Warning: Some NaN data remains ({a:.2f}%). Further cleaning needed.")

print("Cleaning and verification done.")
print("Nan check: ", sales.isnull().sum())

#-----------------------------------------------------------------------------#

##------ Part 1 : Finding best-selling products and sales peaks based on the rating count

"""
Features: 
    
- product_name: Name of the product

Target:
    
- rating_count: Number of people rating a product

"""

# Sort product votes from the most to least voted

sales_sorted_vote = sales.sort_values(by='rating_count', ascending=False)


# Graph with bars

plt.figure(figsize=(12, 6))

# Shortening the name of the products

sales_sorted_vote['product_name'] = sales_sorted_vote['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:50] + '...')

# Differencing most to least voted

c = ['black' if i > 3 else 'purple' for i in range(50)]

# Sorting names and rating counts (lollipop graph)
  
plt.hlines(
    sales_sorted_vote.iloc[0:50]['product_name'],
    sales_sorted_vote.iloc[0:50]['rating_count'],
    xmax=0,
    color=c,
    alpha=0.5
)

# Titling and naming

plt.ylabel("Product's names")
plt.xlabel("Rating counts")
plt.title("Best-selling product, based on rating counts")


plt.tight_layout() # Adjusting the display

plt.show() # Show the graph

# Showing best-selling products

print('The top three best-selling products are :', sales_sorted_vote.iloc[:3]['product_name'])

# Observation and Interpretation

print("Observation : Based on the number of rating count,")  
print("We can see that computer and TV cables have a very high number of rating count.")
print("Interpretation : Customers seemed to prefer HDMI cables")
print("and other technological products, that they may obtain at low cost on Amazon.")

#-----------------------------------------------------------------------------#

##------ Part 2 : Observing whether more expensive products sell less

"""

Features: 
    
- rating_count

Target:
    
- actual price

"""

# Sort product prices from most to least expensive

sales_sorted_price = sales.sort_values(by='actual_price', ascending=False)


# Creation of two graphs

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Selection of cheap and expensive products

expensive = sales_sorted_price.iloc[:20]
cheap = sales_sorted_price.iloc[-20:]


expensive['product_name'] = expensive['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:15] + '...') 

cheap['product_name'] = cheap['product_name'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:15] + '...') 

# Colormap with Seaborn

c0 = sns.cubehelix_palette(as_cmap=True)
c1 = sns.cubehelix_palette(as_cmap=True)

colors = np.random.rand(20) # Value for colormap


# Graph of expensive products

axes[0].hlines(y=expensive['product_name'], xmin=0, xmax=expensive['rating_count'], color='grey')
axes[0].scatter(expensive['rating_count'], expensive['product_name'], c= colors, cmap=c0, marker="^")  # Ajout des points

axes[0].set_xlabel("Rating count")
axes[0].set_title("Expensive products")

# Graph of cheap products

axes[1].hlines(y=cheap['product_name'], xmin=0, xmax=cheap['rating_count'], color='grey')
axes[1].scatter(cheap['rating_count'], cheap['product_name'], c= colors, cmap=c1, marker="v")  # Ajout des points

axes[1].set_xlabel("Rating count")
axes[1].set_title("Cheap products")

plt.title("Comparing the sale of expensive and cheap products")

plt.tight_layout()
plt.show()


# Observation and Interpretation

print("Observation : Based on the number of vote,")  
print("We can see that only two expensive products have a higher rating than")
print("the cheap products. But most of expensive products have a low ",
      "rating than the cheap ones.")
print("Interpretation : Customers seemed to prefer cheaper products than expensive ones,")
print("except when it comes to smartphones and TVs.")


#-----------------------------------------------------------------------------#

##------ Part 3 : Finding which category sells the most and gets the most votes

"""

Features: 
    
- category

Target:
    
- rating count

"""

# Total votes for each category

total = sales.groupby("category")['rating_count'].sum()


# Sorting total Serie

total_sorted_categorymost = total.sort_values(ascending=False) 

#suppress "by=''"


# Transforming "category" to a Dataset

total_sorted_categorymost = total_sorted_categorymost.reset_index()


total_sorted_categorymost['category'] = total_sorted_categorymost['category'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:30] + '...')

# Graph with bars

plt.figure(figsize=(12, 6))

c = ['blue' if i > 1 else 'red' for i in range(25)]

plt.hlines(
    total_sorted_categorymost['category'],
    total_sorted_categorymost['rating_count'],
    xmax=0,
    color=c,
    alpha=0.5
    )

plt.title("Best-selling and most-voted categories")
plt.xlabel("Rating counts")
plt.ylabel("Categories")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# Observation and Interpretation

print("Observation : Based on the number of vote,")  
print("We can see that Electronic is the category the most choosen by customers.")
print("Especially, everything about ", total_sorted_categorymost.iloc[:3]['category'])
print("While the Car category is the least chosen, with only : ", 
      total_sorted_categorymost.iloc[-1]['rating_count'], "votes")
print("Interpretation : This confirms that customers have a preference for electronics ")
print("products, specifically : Headphones/Earbuds, computer accessories, mobile accessories and TV accessories.")


#-----------------------------------------------------------------------------#

##------ Part 4 : Conclusion with a visualization on Seaborn


sales['category'] = sales['category'].apply(lambda x: x 
                                    if len(x) <= 10 else x[:20] + '...')


sns.catplot(x=sales['rating_count'].iloc[:25:2].sort_values(ascending=False), 
                y=sales['actual_price'],data=sales, hue='category')

sns.catplot(x=sales['rating_count'].iloc[:25:2].sort_values(ascending=False), 
                y=sales['rating'],data=sales, hue='category')


print("Final observation : Based on the number of vote and the price,")  
print("We can see that, the more a product is cheap, the more likely it is to be chosen by customers.")
print("Especially, electronics' categories with a high rating vote and a low price.")
print("Moreover, we can observe that the same category has high rate and the most vote.")
print("Conclusion : This confirms that customers have a preference for electronics category on Amazon. ")
print("Specifically : Headphones/Earbuds, computer accessories, mobile accessories and TV accessories,")
print("every accessories they can have at a low price online and be satisfied with the quality of the product.")
















































