import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Load the dataset
file_path = r'C:\Users\manoj\OneDrive\Desktop\DATA ANALYTICS\Final year Project\bothEDMdata2.csv'
df = pd.read_csv(file_path)

df.columns = ['Run Order', 'Pulse On', 'Pulse Off',
              'Wire Feed Rate', 'Voltage', 'Cutting Speed',
              'Kerf Width', 'SR', 'MRR']

# Define input features and target
features = ['Pulse On', 'Pulse Off', 'Wire Feed Rate', 'Voltage', 'Cutting Speed', 'Kerf Width', 'SR']
target = 'MRR'
X = df[features]
y = df[target]

# Add a constant for intercept in regression
X_with_intercept = sm.add_constant(X)

# Perform Linear Regression using statsmodels to get coefficients
lr_model_sm = sm.OLS(y, X_with_intercept).fit()
coefficients = lr_model_sm.params

# Extract coefficients
beta_0 = coefficients['const']
beta_1 = coefficients['Pulse On']
beta_2 = coefficients['Pulse Off']
beta_3 = coefficients['Wire Feed Rate']
beta_4 = coefficients['Voltage']

X = [Pulse_On, Pulse_Off, Wire_Feed, Voltage];
Y = MRR;

X2 = [X, X^2, X(:,1).*X(:,2), X(:,1).*X(:,3), X(:,1).*X(:,4), X(:,2).*X(:,3), X(:,2).*X(:,4), X(:,3).*X(:,4)];
b = regress(Y, [ones(size(X2,1),1), X2])
