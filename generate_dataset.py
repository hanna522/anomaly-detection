import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate the dataset
def generate_dataset():
    start_date = datetime.now()
    date_rng = pd.date_range(start_date, periods=30*24*6, freq='10min')
    data = {
        'timestamp': date_rng,
        'room': np.random.randint(1, 8, size=(len(date_rng))),
        'motion': np.random.randint(0, 2, size=(len(date_rng))),
        'temperature': np.random.uniform(18, 25, size=(len(date_rng))),
        'light': np.random.randint(0, 2, size=(len(date_rng))),
    }
    df = pd.DataFrame(data)
    df['anomaly'] = np.where((df['light'] == 1) & (df['motion'] == 0), 1, 0)
    return df

df = generate_dataset()
df.to_csv('anomaly_detection_dataset.csv', index=False)