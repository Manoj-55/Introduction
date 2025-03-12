import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

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

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_lr = lr_model.predict(X)

# 2. Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X, y)
y_pred_rf = rf_model.predict(X)

# 3. XGBoost
xgb_model = XGBRegressor(random_state=42, n_estimators=100)
xgb_model.fit(X, y)
y_pred_xgb = xgb_model.predict(X)

# Function to print performance metrics
def print_performance(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print("----------------")

# Evaluate performance for all models
print_performance(y, y_pred_lr, "Linear Regression")
print_performance(y, y_pred_rf, "Random Forest")
print_performance(y, y_pred_xgb, "XGBoost")

# Plotting the results
plt.figure(figsize=(12, 8))

# Linear Regression
plt.subplot(3, 1, 1)
plt.scatter(y, y_pred_lr, color='blue', label='Linear Regression')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Linear Regression')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

# Random Forest Regression
plt.subplot(3, 1, 2)
plt.scatter(y, y_pred_rf, color='green', label='Random Forest Regression')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Random Forest Regression')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

# XGBoost Regression
plt.subplot(3, 1, 3)
plt.scatter(y, y_pred_xgb, color='purple', label='XGBoost Regression')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('XGBoost Regression')
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

# Residual plots for each model
plot_residuals(y, y_pred_lr, "Linear Regression")
plot_residuals(y, y_pred_rf, "Random Forest")
plot_residuals(y, y_pred_xgb, "XGBoost")

