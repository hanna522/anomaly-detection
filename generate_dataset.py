import pandas as pd
import numpy as np

# Set the time range
timestamps = pd.date_range(start='2024-01-01', end='2024-01-30', freq='10min')

# Initialize a list to store DataFrames for each room
dfs = []

# Generate data for 5 rooms
for room in range(1, 6):
    # Initialize motion, light, and temperature arrays based on the length of timestamps
    motion = np.zeros(len(timestamps))
    light = np.zeros(len(timestamps))
    temperature = np.zeros(len(timestamps))

    # Generate motion, light, and temperature values based on specified conditions
    for i, timestamp in enumerate(timestamps):
        if (timestamp.weekday() >= 5) or (timestamp.weekday() < 5 and (timestamp.hour < 9 or timestamp.hour >= 19)):
            motion[i] = 0
            light[i] = 0
        else:
            motion[i] = np.random.choice([0, 1])
            light[i] = 1 if motion[i] == 1 else np.random.choice([0, 1])
        
        # Generate temperature randomly within a plausible range for indoor environments
        temperature[i] = np.random.uniform(18, 25)

    # Create a DataFrame with the generated data for this room
    df = pd.DataFrame({
        'timestamp': timestamps,
        'room': room,
        'motion': motion,
        'light': light,
        'temperature': temperature
    })

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
df_all_rooms = pd.concat(dfs)

# Save the DataFrame to a CSV file
df_all_rooms.to_csv('anomaly_detection_dataset.csv', index=False)

print("Successfully generated dataset!")
