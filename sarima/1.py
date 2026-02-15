import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('Book1.csv')
if 'Grand Total' in df.columns:
    df.drop(columns=['Grand Total'], inplace=True)

# 2. Fix Dates
def parse_dates_fix_years(date_series, start_year=2021):
    dates = []
    current_year = start_year
    prev_month = -1
    temp_dates = pd.to_datetime(date_series, format='%d-%b', errors='coerce')
    
    for date in temp_dates:
        if pd.isna(date):
            dates.append(pd.NaT)
            continue
        month = date.month
        if prev_month != -1 and month < prev_month:
            current_year += 1
        dates.append(datetime(current_year, month, date.day))
        prev_month = month
    return dates

df['Date'] = parse_dates_fix_years(df['Row Labels'])
df = df.dropna(subset=['Date']).set_index('Date').drop(columns=['Row Labels']).sort_index()

# 3. Filter Stable Sensors (< 30% missing)
threshold = 0.3
missing_percent = df.isnull().mean()
stable_sensors = missing_percent[missing_percent < threshold].index.tolist()
print(f"Stable Sensors: {stable_sensors}")

df_stable = df[stable_sensors].interpolate(method='time').bfill().ffill()

# 4. Train and Test Loop
results = {}
lags = [1, 2, 3, 7] # Use past 1, 2, 3, and 7 days to predict today

for sensor in stable_sensors:
    # Create lag features
    data = pd.DataFrame(df_stable[sensor])
    for lag in lags:
        data[f'lag_{lag}'] = data[sensor].shift(lag)
    data.dropna(inplace=True)
    
    # Split 80/20
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    X_train, y_train = train.drop(columns=[sensor]), train[sensor]
    X_test, y_test = test.drop(columns=[sensor]), test[sensor]
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[sensor] = rmse
    print(f"Sensor {sensor} RMSE: {rmse:.2f}")

    # Plotting (Optional example for one sensor)
    if sensor == '2':
        plt.figure(figsize=(10, 5))
        plt.plot(test.index, y_test, label='Actual')
        plt.plot(test.index, preds, label='Predicted', linestyle='--')
        plt.title(f'Sensor {sensor} Prediction')
        plt.legend()
        plt.show()