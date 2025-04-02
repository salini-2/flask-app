import pandas as pd

# Load the existing hospital and blood bank data
file_path = "hospitals_bloodbanks_real.csv"  # Ensure this file is in the same directory
df_hospitals = pd.read_csv(file_path)

# Load the historical blood demand data
history_path = "blood_demand_history.csv"  # Ensure this file exists
df_demand = pd.read_csv(history_path)

# Merge both datasets on Hospital Name and Blood Type
df_merged = df_hospitals.merge(df_demand, on=["Name", "Blood Type"], how="left")

# Save the merged data as a new file
merged_file_path = "updated_hospitals_bloodbanks.csv"
df_merged.to_csv(merged_file_path, index=False)

print(f"✅ Merged file saved as {merged_file_path}")
