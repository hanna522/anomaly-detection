from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Hugging Face
dataset = load_dataset("hangyeol522/anomaly-detection-model")

# Convert to pandas dataframe
df = dataset['train'].to_pandas()

# Prepare the data
X = df[['room', 'motion', 'temperature', 'light']]
y = df['anomaly']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Visualize the results
sns.countplot(x=y_pred)
plt.title('Anomaly Detection Results')
plt.show()
