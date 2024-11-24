
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump, load

# Loading Data
df = pd.read_csv("Advertising.csv")

# Preparing input (X) and output (y)
X = df.drop('sales',axis=1)
y = df['sales']

# Generating Polynomial features from X (input)
polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
poly_features = polynomial_converter.fit_transform(X)

# Preparing dataframe using polynomial features - this is optional step
poly_feature_names = list(polynomial_converter.get_feature_names_out())
poly_features_df = pd.DataFrame(poly_features,columns=poly_feature_names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(poly_features_df, y, test_size=0.3, random_state=101,)

# Preparing Model - Need to include hyper-parameters at this stage
model = LinearRegression(fit_intercept=True)

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


# Emperical Formula to interpret the model
# y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x1^2 + b5*x1*x2 + b6*x1*x3 + b7*x2^2 + b8*x2*x3 + b9*x3^2
# Ex: sales = 5.12555742313269 + 0.0517095811*TV + 0.0130848864*radio + 0.0120000085*newspaper -0.000110892474*TV^2 + ...

###### -------Train the model on Full data --------- ###### 
final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)
final_model = LinearRegression()

final_model.fit(final_poly_converter.fit_transform(X),y)

campaign = [[149,22,12]]
y_new_predict = final_model.predict(campaign)


# Save the model
dump(final_model, 'sales_model.joblib') 
# Save polynomial converter - needed to fit new samples
dump(final_poly_converter,'poly_converter.joblib')

###### ------- Applying pre-trained model to new samples --------- ###### 
# Loading polynomial converter
loaded_poly = load('poly_converter.joblib')
# Loading model 
loaded_model = load('sales_poly_model.joblib')

campaign = [[149,22,12]]
campaign_poly = loaded_poly.transform(campaign)
final_model.predict(campaign_poly)
