import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate the dataset 
# room: 1-5, motion: 0/1, temperature: 18-30(C), light: 0/1, time stamp: every 10min, 30 days
def generate_dataset():
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 6, 30, 23, 59)
    date_rng = pd.date_range(start_date, end_date, freq='10min')
    data = []
    
    for room in range(1, 6):
      room_data = {
          'timestamp': date_rng,
          'room': room,
          'motion': np.random.randint(0, 2, size=(len(date_rng))),
          'temperature': np.random.uniform(18, 30, size=(len(date_rng))),
          'light': np.random.randint(0, 2, size=(len(date_rng))),
      }
      room_df = pd.DataFrame(room_data)
      room_df['anomaly'] = np.where((room_df['light'] == 1) & (room_df['motion'] == 0), 1, 0) # for checking accuracy
      data.append(room_df)
      
    df = pd.concat(data, ignore_index=True)
    return df

df = generate_dataset()
df.to_csv('anomaly_detection_dataset.csv', index=False)