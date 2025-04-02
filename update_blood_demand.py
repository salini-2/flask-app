import pandas as pd
import numpy as np
import os

# Load existing CSV file
csv_path = "hospitals_bloodbanks_real.csv"
updated_csv_path = "updated_hospitals_bloodbanks.csv"

df = pd.read_csv(csv_path)

# Ensure necessary columns exist
required_columns = {'Name', 'Latitude', 'Longitude', 'Blood Type', 'Availability (Units)'}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV file is missing required columns")

# Generate synthetic past demand data with unique values
demand_data = []
np.random.seed(42)  # Ensures reproducibility

hospital_base_demand = {}  # Store unique demand for each hospital

for _, row in df.iterrows():
    hospital_name = row['Name']
    blood_type = row['Blood Type']
    
    # Ensure each hospital has a unique base demand range
    if hospital_name not in hospital_base_demand:
        hospital_base_demand[hospital_name] = np.random.randint(50, 200)
    
    base_demand = hospital_base_demand[hospital_name] + np.random.randint(-10, 10)
    trend_factor = np.random.uniform(0.98, 1.05)  # Unique small trend variations

    for i in range(1, 31):  # Generate 30 days of past demand
        unique_demand = int(base_demand * trend_factor + np.random.randint(-10, 10))
        unique_demand = max(unique_demand, 20)  # Ensure minimum demand of 20 units
        
        demand_data.append({
            "Date": pd.Timestamp(f"2024-01-{i:02d}"),
            "Hospital": hospital_name,
            "Blood Type": blood_type,
            "Demand": unique_demand
        })

# Convert to DataFrame
df_demand = pd.DataFrame(demand_data)

# Save updated CSV with unique values
if os.path.exists(updated_csv_path):
    os.remove(updated_csv_path)  # Remove if already exists
df_demand.to_csv(updated_csv_path, index=False)

print(f"Updated file saved as {updated_csv_path}")
