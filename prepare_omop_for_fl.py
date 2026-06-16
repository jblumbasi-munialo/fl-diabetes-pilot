# prepare_omop_for_fl.py
from omop_loader import load_omop_data

# For node_A (different database or same database with person filtering – adapt)
df_A = load_omop_data("omop_diabetes_A.db")   # if you have separate DBs per node
df_A.to_csv("data/node_A/diabetes.csv", index=False)

df_B = load_omop_data("omop_diabetes_B.db")
df_B.to_csv("data/node_B/diabetes.csv", index=False)

print("CSV files created for both nodes.")
