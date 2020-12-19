import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import computeCost as cc
import gradientDescent as gd


housing = pd.read_csv("housing.csv")

#-------------------Dropping NaN value-------------------
print("데이터에 속한 NaN의 개수")
print(housing.isnull().sum())
print()

housing = housing.dropna(axis=0)#Dropping NaN value

print("Drop 수행 후 데이터에 속한 NaN의 개수")
print(housing.isnull().sum())
print()


#-------------------Plotting-------------------
print("데이터에 대한 정보")
print(housing.describe())
print()



housing.plot(kind="scatter", x="longitude", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="latitude", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="housing_median_age", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="total_rooms", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="total_bedrooms", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="population", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="households", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
housing.plot(kind="scatter", x="ocean_proximity", y="median_house_value", alpha=0.1)

plt.show()

#-------------------Data-------------------

#Data
x_data = housing.drop(housing.columns[[8]], axis = 1) # Dropping ['median_house_value']
y_data = housing['median_house_value']

#Data separation

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=123)

#Data scaling
print("데이터 스케일링 전")
print(x_train.head())
print()

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = pd.DataFrame(data = scaler.transform(x_train), columns = x_train.columns, index= x_train.index)
x_test = pd.DataFrame(data = scaler.transform(x_test), columns = x_test.columns, index= x_test.index)

print("데이터 스케일링 후")
print(x_train.head())
print()

#행렬연산을 위해 데이터를 numpy의 행렬 형태로 변환
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#-------------------Linear Regresssion-------------------
model = LinearRegression().fit(x_train, y_train)

x_test_predict = model.predict(x_test)

for i in range(x_test_predict.shape[0]): #500001초과는 500001로 고정
    if x_test_predict[i] > 500001:
        x_test_predict[i] = 500001


lin_mse = mean_squared_error(y_test, x_test_predict)
lin_rmse = np.sqrt(lin_mse)
print("sklearn에서 제공하는 선형회귀 함수를 이용하여 구한 Cost")
print("RMSE: ", lin_rmse)
print()

#-------------------Own Linear Regresssion-------------------

#Bias term 삽입
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

theta = np.array([int(random.random()*10) for i in range(9+1)])

theta, J = gd.gradientDescent(x_train, y_train, theta, 0.1, 3000)
print("직접 구현한 선형회귀 함수를 이용하여 구한 Cost")
print("RMSE: ", np.sqrt(cc.computeCost(x_test, y_test, theta)))
