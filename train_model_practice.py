from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
data = dataset['train'].to_pandas()  # Load the training portion of the dataset

# Select the necessary features
features = ['room', 'motion', 'temperature', 'light']
X = data[features]

# Create and train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Predict and add results
data['anomaly'] = model.predict(X)
data['anomaly'] = data['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Group anomalies by room and hourly time period
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour  # Extract hour from timestamp
data['time_interval'] = data['hour'].apply(lambda x: f'{x:02d}:00 - {x+1:02d}:00')
anomalies = data[data['anomaly'] == 1]

# Aggregate the number of anomalies by room and time interval
anomalies_count = anomalies.groupby(['time_interval', 'room']).size().unstack(fill_value=0)

# Visualize the number of anomalies by room and time interval using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(anomalies_count.T, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel('Time Interval')
plt.ylabel('Room')
plt.title('Number of Anomalies by Room and Hourly Time Interval')
plt.show()

# Check the anomaly data
print(anomalies)

# Save the anomaly data
anomalies.to_csv('detected_anomalies.csv', index=False)
