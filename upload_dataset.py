from datasets import Dataset
import pandas as pd

# Load your dataset
df = pd.read_csv('anomaly_detection_dataset.csv')

# Convert pandas dataframe to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Push the dataset to Hugging Face
dataset.push_to_hub("hangyeol522/anomaly-detection-model")

print("Successfully uploaded!")