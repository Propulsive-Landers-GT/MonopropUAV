import pandas as pd
import numpy as np

# Load the CSV file
# (replace 'data.csv' with your actual filename)
df = pd.read_csv('trajectory_data.csv')

# Ensure the columns exist
if 'Time' not in df.columns or 'Thrust_Mag' not in df.columns:
    raise ValueError("CSV must contain 'Time' and 'Thrust_Mag' columns")

# Extract time and thrust as numpy arrays
time = df['Time'].to_numpy()
thrust = df['Thrust_Mag'].to_numpy()

# Compute delta t from first two entries
if len(time) < 2:
    raise ValueError("Not enough time entries to compute delta t")

delta_t = time[1] - time[0]
print(f"Δt (from first two samples): {delta_t}")

# Integrate thrust over time using the trapezoidal rule
impulse = np.trapz(thrust, time)
print(f"Integrated Thrust (Impulse): {impulse}")
