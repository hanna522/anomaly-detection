from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
df = dataset['train'].to_pandas() # Load the training portion of the dataset

# Prepare the data
X = df[['room', 'motion', 'temperature', 'light']]
y = df['anomaly']

# Add a column for hourly time group
df['time'] = pd.to_datetime(df['timestamp']).dt.time
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['time_group'] = df['hour'].apply(lambda x: f"{x}:00 - {x+1}:00")

# Define the correct order for the time groups
time_groups = [f"{i:02d}:00 - {i+1:02d}:00" for i in range(24)]
df['time_group'] = pd.Categorical(df['time_group'], categories=time_groups, ordered=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Add predictions to the test set
df_test = df.iloc[X_test.index]
df_test['predicted_anomaly'] = y_pred

# Evaluate the performance
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Group by room and hourly time group
grouped = df_test.groupby(['room', 'time_group'])['predicted_anomaly'].sum().reset_index()

# Display the grouped data
print(grouped)

# Visualize the results by room and hourly time group
plt.figure(figsize=(12, 6))
sns.barplot(x='room', y='predicted_anomaly', hue='time_group', data=grouped)
plt.title('Anomaly Detection Results by Room and Hourly Time Group')
plt.xlabel('Room')
plt.ylabel('Number of Anomalies')
plt.legend(title='Time Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
