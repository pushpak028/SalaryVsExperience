import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)

lr = LinearRegression()
lr.fit(X_train,Y_train)


y_pred = lr.predict(X_test)

plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_test , y_pred , color = 'blue')
plt.title("salary vs experience")
plt.xlabel("exp")
plt.ylabel("salary")
plt.show()

