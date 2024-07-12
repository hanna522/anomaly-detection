from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
df = dataset['train'].to_pandas()

# Define the true anomaly condition
df['true_anomaly'] = ((df['light'] == 1) & (df['motion'] == 0)).astype(int)

# Select features
X = df[['light', 'motion']]
y = df['true_anomaly']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model (only using features, no labels)
model = IsolationForest(contamination=0.2)
model.fit(X_train)

# Predict anomalies on the test set
y_pred = model.predict(X_test)
y_pred = [0 if x == 1 else 1 for x in y_pred]  # 1: anomaly, 0: normal

# Evaluate the model using true labels
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Add predictions to the test dataframe for analysis and visualization
df_test = X_test.copy()
df_test['anomaly'] = y_pred

# Add the 'room' column from the original dataframe
df_test['room'] = df.loc[X_test.index, 'room']

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