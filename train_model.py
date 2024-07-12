from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
df = dataset['train'].to_pandas()

# Select features
X = df[['light', 'motion']]

# Split the dataset into training and test sets
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.2)
model.fit(X_train)

# Predict anomalies on the test set
df_test['anomaly'] = model.predict(X_test)
df_test['anomaly'] = df_test['anomaly'].map({1: 0, -1: 1})  # 1: normal, -1: anomaly

# Count the number of anomalies for each room in the test set
anomalies_per_room = df_test[df_test['anomaly'] == 1].groupby('room').size()

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar plot with a fixed x-axis label 'room = 1'
anomalies_per_room.plot(kind='bar', ax=ax, color='red', alpha=0.7)
ax.set_xlabel('Room')
ax.set_ylabel('Number of Anomalies')
ax.set_title('Number of Anomalies Detected per Room')
ax.set_xticklabels(['Room 1', 'Room 2', 'Room 3', 'Room 4', 'Room 5'])

plt.show()

# Save the test dataframe with anomaly predictions to a CSV file
df_test.to_csv('anomaly_detection_results.csv', index=False)
