# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection & Preparation

Import the dataset containing hours studied and marks scored.

Split the dataset into input (X = hours) and output (Y = marks).

2.Train–Test Split

Divide the dataset into training data and testing data.

3.Model Training

Apply the Simple Linear Regression algorithm on the training data to learn the relationship between hours studied (X) and marks scored (Y).

4.Prediction & Evaluation

Use the trained model to predict marks for the test data.

Compare predicted values with actual values to evaluate accuracy.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MALENI M
RegisterNumber:  212223040110

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```

<img width="501" height="245" alt="Screenshot 2025-08-18 115009" src="https://github.com/user-attachments/assets/3b42d51d-7666-47bd-9cdf-665ac23b5339" />

```
df.tail()

```
<img width="498" height="239" alt="Screenshot 2025-08-18 115126" src="https://github.com/user-attachments/assets/cd488dfe-02f3-4a91-ac1b-25c0970ca75c" />

```
x=df.iloc[:,:-1].values
x
```
<img width="918" height="600" alt="Screenshot 2025-08-18 115225" src="https://github.com/user-attachments/assets/f164c0f2-0883-4d20-9876-21df40b52b07" />

```
y=df.iloc[:,1].values
y
```
<img width="973" height="55" alt="Screenshot 2025-08-18 115330" src="https://github.com/user-attachments/assets/72ba2ffa-4ade-4695-9d02-ce04f397bcbe" />

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```
<img width="908" height="53" alt="Screenshot 2025-08-18 115430" src="https://github.com/user-attachments/assets/b7271d05-5844-4add-b1c0-077ba2ec2439" />

```
Y_test
```
<img width="828" height="38" alt="Screenshot 2025-08-18 115522" src="https://github.com/user-attachments/assets/13a8c964-da87-4e13-b2b2-1110c3e1cd16" />

```
print("Register Number: 212223040110")
print("Name: MALENI M")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Register Number: 212223040110")
print("Name: MALENI M")
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,Y_pred,color="red")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Register Number: 212223040110")
print("Name: MALENI M")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

<img width="1920" height="1080" alt="Screenshot 2025-08-20 080338" src="https://github.com/user-attachments/assets/a4f9e82d-565a-4686-8d8a-d58e0061b520" />
<img width="1920" height="1080" alt="Screenshot 2025-08-20 080400" src="https://github.com/user-attachments/assets/a4b7fb09-3d72-4e90-814b-78d26c57dbaf" />
<img width="1920" height="1080" alt="Screenshot 2025-08-20 080411" src="https://github.com/user-attachments/assets/7a2edc4d-9c01-4b4c-a373-5d738ead0776" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
