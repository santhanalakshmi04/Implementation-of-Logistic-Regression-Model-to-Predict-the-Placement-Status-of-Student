# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Santhana Lakshmi k
RegisterNumber: 212222240091
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
1.PLACEMENT DATA:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/ee33d2ba-a53d-498f-85b8-e2dab2cdbedb)

2.SALARY DATA:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/8725adf4-c455-422c-a372-2dbb1e8113c4)

3.CHECKING THE NULL() FUNCTION:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/cb4fb35a-2bb8-493c-bae3-5e3121976a34)

4.DATA DUPLICATE:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/d3cccb38-4652-4180-8ff6-4706075d56c6)

5.PRINT DATA:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/56730a28-df28-41c7-bda7-67e3d37a229b)

6.DATA STATUS:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/0c4a453f-2c31-4c96-9592-03932695c344)

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/58ea4c7a-363d-4ba5-802e-d23817a8bd16)


7.Y_PREDICATION ARRAY:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/0fbee6ad-a702-4921-9de9-a151a6632a66)

8.ACCURACY VALUE:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/f694719f-4dc5-4f9b-bb40-6cf738238833)

9.CONFUSION ARRAY:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/0c4a235c-e8a8-4216-ad4a-39201be2bfb9)

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/f8398a01-728e-4203-9cc7-02722a726f50)

10.CLASSIFICATION REPORT:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/8837b5af-b14c-4184-9385-8dc065da0c41)

PREDICTION OF LR:

![image](https://github.com/preethi2831/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155142246/b1511eef-9954-49c5-8739-24ba5cac5496)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
