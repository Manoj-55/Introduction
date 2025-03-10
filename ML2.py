import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Create polynomial features for RSM
X = X.copy()
for col in X_columns:
    X[f'{col}^2'] = X[col]**2
for pair in product(X_columns, repeat=2):
    if pair[0] != pair[1]:
        X[f'{pair[0]}*{pair[1]}'] = X[pair[0]] * X[pair[1]]

# Fit RSM model
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())

# Function to generate and plot 3D surfaces
def plot_surface(x_var, y_var, title):
    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 50)
    y_range = np.linspace(df[y_var].min(), df[y_var].max(), 50)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    Z_mesh = np.zeros_like(X_mesh)
    
    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            params = {var: np.mean(df[var]) for var in X_columns}  # Set all to mean
            params[x_var] = X_mesh[i, j]
            params[y_var] = Y_mesh[i, j]
            
            # Compute polynomial and interaction terms dynamically
            param_values = [params[var] for var in X_columns]
            poly_terms = [params[var]**2 for var in X_columns]
            interaction_terms = [params[pair[0]] * params[pair[1]] for pair in product(X_columns, repeat=2) if pair[0] != pair[1]]
            x_vals = np.array([1] + param_values + poly_terms + interaction_terms)
            Z_mesh[i, j] = model.predict(x_vals.reshape(1, -1))[0]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='jet', edgecolor='none')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(y_column)
    plt.title(title)
    plt.show()

# Generate multiple surface plots
plot_surface('Pulse On', 'Pulse Off', 'MRR vs. (Pulse On, Pulse Off)')
plot_surface('Wire Feed Rate', 'Pulse On', 'MRR vs. (Wire Feed Rate, Pulse On)')
plot_surface('Wire Feed Rate', 'Pulse Off', 'MRR vs. (Wire Feed Rate, Pulse Off)')
plot_surface('Voltage', 'Pulse On', 'MRR vs. (Voltage, Pulse On)')
plot_surface('Voltage', 'Pulse Off', 'MRR vs. (Voltage, Pulse Off)')
plot_surface('Voltage', 'Wire Feed Rate', 'MRR vs. (Voltage, Wire Feed Rate)')

# Machine Learning Models
features = ['Pulse On', 'Pulse Off', 'Wire Feed Rate', 'Voltage', 'Cutting Speed', 'Kerf Width']
X_ml = df[features]
y_ml = df[y_column]

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_ml, y_ml)
y_pred_lr = lr_model.predict(X_ml)

# Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_ml, y_ml)
y_pred_rf = rf_model.predict(X_ml)

# XGBoost
xgb_model = XGBRegressor(random_state=42, n_estimators=100)
xgb_model.fit(X_ml, y_ml)
y_pred_xgb = xgb_model.predict(X_ml)

# Function to print performance metrics
def print_performance(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print("---------------------------")

# Evaluate performance for all models
print_performance(y_ml, y_pred_lr, "Linear Regression")
print_performance(y_ml, y_pred_rf, "Random Forest")
print_performance(y_ml, y_pred_xgb, "XGBoost")

# Plot results
plt.figure(figsize=(12, 8))

models = [(y_pred_lr, 'Linear Regression', 'blue'),
          (y_pred_rf, 'Random Forest', 'green'),
          (y_pred_xgb, 'XGBoost', 'purple')]

for i, (y_pred, title, color) in enumerate(models, 1):
    plt.subplot(3, 1, i)
    plt.scatter(y_ml, y_pred, color=color, label=title)
    plt.plot([min(y_ml), max(y_ml)], [min(y_ml), max(y_ml)], color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

plt.tight_layout()
plt.show()

# Residual analysis
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'{model_name} Residuals')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()

# Residual plots
for y_pred, title, _ in models:
    plot_residuals(y_ml, y_pred, title)
