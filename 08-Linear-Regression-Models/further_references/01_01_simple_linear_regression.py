
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump, load

# Loading Data
df = pd.read_csv("Advertising.csv")

# Preparing input (X) and output (y)
X = df['TV'] # Selecting one feature as X
y = df['sales']

# X contains 1 feature - 'TV', hence it is simple linear regression
print(X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101,)

# Preparing Model - Need to include hyper-parameters at this stage
model = LinearRegression()

# Training Model
model.fit(X_train,y_train)

# Test Model - On Test dataset
test_predictions = model.predict(X_test)

# Evaluating Metrics
MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

# Intercept Model -- Explainability of the ML model
print(model.intercept_)  # b0
print(model.coef_)
print(model.feature_names_in_)

## Emperical Formula to interpret the model
# y = b0 + b1*x
# Ex: sales = 3.151526768070651 + 0.04469599*tv

# Train the model on Full Data
model.fit(X,y)
y_new_predict = model.predict(X_new)

# Save the model
dump(final_model, 'sales_model.joblib') 

## Emperical Formula to build using 
# y = b0 + b1*x
# Ex: sales = 3.151526768070651 + 0.04469599*tv 


