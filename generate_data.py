import pandas as pd
import numpy as np
import os

def generate_diabetes_data(n_samples=500, seed=42):
    np.random.seed(seed)
    data = {
        'age': np.random.normal(55, 12, n_samples),
        'bmi': np.random.normal(28, 6, n_samples),
        'glucose': np.random.normal(140, 40, n_samples),
        'blood_pressure': np.random.normal(80, 12, n_samples),
        'insulin': np.random.normal(80, 50, n_samples),
        'pregnancies': np.random.poisson(2, n_samples),
        'skin_thickness': np.random.normal(30, 8, n_samples),
        'diabetes_pedigree': np.random.exponential(0.5, n_samples),
    }
    df = pd.DataFrame(data)
    # Simple rule: glucose > 150 AND bmi > 30 -> diabetes risk
    risk = ((df['glucose'] > 150) & (df['bmi'] > 30)).astype(int)
    df['target'] = risk
    return df

# Create directories
os.makedirs('data/node_A', exist_ok=True)
os.makedirs('data/node_B', exist_ok=True)

# Generate two slightly different datasets for two hospitals
df_A = generate_diabetes_data(500, seed=42)
df_B = generate_diabetes_data(500, seed=123)

df_A.to_csv('data/node_A/diabetes.csv', index=False)
df_B.to_csv('data/node_B/diabetes.csv', index=False)

print("Datasets created in data/node_A/ and data/node_B/")
