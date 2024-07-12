from sklearn.ensemble import IsolationForest
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
df = dataset['train'].to_pandas()  # Load the training portion of the dataset

# Select features
X = df[['light', 'motion']]

# Train the Isolation Forest model
model = IsolationForest(contamination=0.2)
model.fit(X)

# Predict anomalies
df['anomaly'] = model.predict(X)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1: normal, -1: anomaly

# Count the number of anomalies for each room
anomalies_per_room = df[df['anomaly'] == 1].groupby('room').size()

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar plot with a fixed x-axis label 'room = 1'
anomalies_per_room.plot(kind='bar', ax=ax, color='red', alpha=0.7)
ax.set_xlabel('Room')
ax.set_ylabel('Number of Anomalies')
ax.set_title('Number of Anomalies Detected per Room')
ax.set_xticklabels(['Room 1', 'Room 2', 'Room 3', 'Room 4', 'Room 5'])

plt.show()

# Save the dataframe to a CSV file
df.to_csv('anomaly_detection_results.csv', index=False)