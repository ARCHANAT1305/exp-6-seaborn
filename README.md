# EXNO-6 DATA VISUALIZATION USING SEABORN LIBRARY
## AIM:
To Perform Data Visualization using seaborn python library for the given datas.

## EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

## ALOGRITHM:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

## CODING AND OUTPUT:
### NAME : ARCHANA T
### REGISTER NUMBER : 212223240013
```

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Line Plot - Basic
x = [1, 2, 3, 4, 5]
y = [3, 6, 2, 7, 1]

sns.set(style="whitegrid")
sns.lineplot(x=x, y=y)
plt.title("Seaborn Line Plot")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

# Line Plot - Multiple Lines
x = [1, 2, 3, 4, 5]
y1 = [3, 5, 2, 6, 1]
y2 = [1, 6, 4, 3, 8]
y3 = [5, 2, 7, 1, 4]

df_multi = pd.DataFrame({
    'X': x * 3,
    'Y': y1 + y2 + y3,
    'Series': ['Y1']*5 + ['Y2']*5 + ['Y3']*5
})

sns.lineplot(data=df_multi, x='X', y='Y', hue='Series')
plt.title("Multiple Lines with Seaborn")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

# Bar Plot - Tips Dataset
tips = sns.load_dataset('tips')

plt.figure(figsize=(8, 5))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title("Total Bill by Day and Gender")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show()


# Titanic Dataset
tit = pd.read_csv("titanic_dataset.csv")

# Barplot: Fare by Embarked Town 
plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked', y='Fare', hue='Embarked', data=tit, palette='rainbow', legend=False)
plt.title("Fare of Passenger by Embarked Town")
plt.xlabel("Embarked Town")
plt.ylabel("Fare")
plt.show()

# Barplot: Fare by Embarked and Pclass
plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked', y='Fare', hue='Pclass', data=tit, palette='rainbow')
plt.title("Fare of Passenger by Embarked Town, Divided by Class")
plt.xlabel("Embarked Town")
plt.ylabel("Fare")
plt.show()


# Scatter Plot - Tips Dataset
plt.figure(figsize=(7, 5))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
plt.grid(True)
plt.show()


# Violin Plot - Fixed palette warning
plt.figure(figsize=(8, 5))
sns.violinplot(data=tit, x='Sex', y='Age', hue='Sex', palette='Set2', legend=False)
plt.title("Age Distribution by Gender (Violin Plot)")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()


# Histogram - Random Marks Data
np.random.seed(0)
marks = np.random.normal(loc=70, scale=10, size=100)

plt.figure(figsize=(8, 5))
sns.histplot(marks, bins=15, kde=True, color='purple')
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()


# Histogram - Titanic Dataset (Survival by Class)
plt.figure(figsize=(8, 5))
sns.histplot(data=tit, x='Pclass', hue='Survived', multiple="stack", kde=True)
plt.title("Passenger Class Distribution by Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()
```
![image](https://github.com/user-attachments/assets/77e09eac-3b36-4ac2-afd8-17e4f5af077d)
![image](https://github.com/user-attachments/assets/8a8a7bcd-4bdc-43b0-9c81-7fe82b50821b)
![image](https://github.com/user-attachments/assets/ea0c91f6-08ca-4f87-990e-d85a0f1d1cd6)
![image](https://github.com/user-attachments/assets/69d4c5ce-91fa-40b9-8ee0-60c786d620b0)
![image](https://github.com/user-attachments/assets/faa43dd4-604a-4274-8f2d-82bedb5de9db)
![image](https://github.com/user-attachments/assets/01e68cf7-5d45-4318-a462-c968ad775572)
![image](https://github.com/user-attachments/assets/ea2af522-f99b-4c2f-8a3a-f478c046098c)
![image](https://github.com/user-attachments/assets/0de4de9c-2a13-48fc-bf3c-d2cd9f82969e)
![image](https://github.com/user-attachments/assets/8772ac06-1b86-4540-b1af-1d344bcbd422)

## RESULT:
Thus the program to Perform Data Visualization using seaborn was executed successfully.
