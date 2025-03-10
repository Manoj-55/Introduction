import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import minimize
from itertools import product

# Load and prepare data
data = pd.read_csv(r'C:\Users\manoj\OneDrive\Desktop\DATA ANALYTICS\Final year Project\bothEDMdata2.csv')
data = data.rename(columns=lambda x: x.strip())

# Experimental parameter levels (from dataset)
PARAM_LEVELS = {
    'Pulse On': [1, 2, 3, 5],
    'Pulse Off': [6, 7, 8, 9],
    'Wire Feed Rate': [3, 4, 5, 6],
    'Voltage': [40, 50, 60, 70]
}


# STEP 1: Build RSM Models

def rsm_formula(response):
    return f"""
    {response} ~ Q('Pulse On') + Q('Pulse Off') + Q('Wire Feed Rate') + Q('Voltage') +
    I(Q('Pulse On')**2) + I(Q('Voltage')**2) +
    Q('Pulse On'):Q('Pulse Off') + Q('Wire Feed Rate'):Q('Voltage')
    """

mrr_model = ols(rsm_formula('MRR'), data=data).fit()
sr_model = ols(rsm_formula('SR'), data=data).fit()


# STEP 2: Continuous Optimization

def objective(x):
    input_df = pd.DataFrame([x], columns=PARAM_LEVELS.keys())
    sr_pred = sr_model.predict(input_df)[0]
    mrr_pred = mrr_model.predict(input_df)[0]
    sr_penalty = 100 * max(0, sr_pred - 2)**3
    return -mrr_pred + sr_penalty

continuous_result = minimize(
    objective,
    x0=[3, 7.5, 4.5, 55],
    bounds=[(min(v), max(v)) for v in PARAM_LEVELS.values()],
    method='L-BFGS-B'
)


# STEP 3: Integer Optimization and Validation 

def round_to_nearest_level(value, levels):
    return min(levels, key=lambda x: abs(x - value))

# Get nearest valid integers
rounded_params = {
    factor: round_to_nearest_level(continuous_result.x[i], levels)
    for i, (factor, levels) in enumerate(PARAM_LEVELS.items())
}

# Create parameter options for each factor
param_options = {}
for factor in PARAM_LEVELS:
    current_value = rounded_params[factor]
    idx = PARAM_LEVELS[factor].index(current_value)
    param_options[factor] = [
        PARAM_LEVELS[factor][max(0, idx-1)],
        current_value,
        PARAM_LEVELS[factor][min(len(PARAM_LEVELS[factor])-1, idx+1)]
    ]

# Generate all possible combinations
param_combinations = product(*[
    [(factor, val) for val in vals]
    for factor, vals in param_options.items()
])

# Evaluate all combinations
valid_results = []
for combo in param_combinations:
    params = {factor: val for factor, val in combo}
    input_df = pd.DataFrame([params.values()], columns=params.keys())
    sr_pred = sr_model.predict(input_df)[0]
    mrr_pred = mrr_model.predict(input_df)[0]
    
    if sr_pred <= 2:
        valid_results.append({
            'params': params,
            'MRR': mrr_pred,
            'SR': sr_pred
        })

# Select best valid combination
if valid_results:
    best_result = max(valid_results, key=lambda x: x['MRR'])
    final_params = best_result['params']
else:
    final_params = rounded_params

# STEP 4: Final Results

print("\nOptimal Rounded Parameters:")
for factor in PARAM_LEVELS:
    print(f"{factor}: {final_params[factor]}")

final_mrr = mrr_model.predict(pd.DataFrame([final_params]))[0]
final_sr = sr_model.predict(pd.DataFrame([final_params]))[0]
print(f"\nPredicted MRR: {final_mrr:.3f}")
print(f"Predicted SR: {final_sr:.3f}")


# STEP 5: ANOVA Tables

print("\nMRR ANOVA Table:")
print(sm.stats.anova_lm(mrr_model, typ=2))
print("\nSR ANOVA Table:")
print(sm.stats.anova_lm(sr_model, typ=2))
