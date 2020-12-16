import xgboost as xgb
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
import argparse

data = pd.read_csv("datasets/all_data_with_negative_features.csv",encoding="utf8")
data_filter = pd.read_csv("datasets/feature12.csv",encoding="utf8")
data = data.sample(frac=1.0)
data.reset_index(drop=True)
# data = data.filter(items=data_filter)
data_x = data.iloc[:,2:-1].values
data_y = np.log10(data.iloc[:,-1].values)

X_train, X_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.15,random_state=0)

model = xgb.XGBRegressor(max_depth=12, learning_rate=0.1, n_estimators=300, objective='reg:squarederror') # squaredlogerror
model.fit(X_train,y_train)
predict_value = model.predict(X_test)

print(mean_squared_error(predict_value,y_test))
print(r2_score(predict_value,y_test))
x = np.arange(-2,6,1)
plt.xlim((-1,4.5))
plt.ylim((-1,4.5))
plt.xlabel("Predict")
plt.ylabel("Origin")
plt.plot(x,x+1,'--',label="y=x+1")
plt.plot(x,x+0,'--',label="y=x")
plt.plot(x,x-1,'--',label="y=x-1")
plt.scatter(predict_value,y_test,marker='o',color = 'blue', s=10)
plt.legend(loc='upper left')
plt.savefig("xgboost.png")
plt.clf()

