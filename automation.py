import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pytz
from dotenv import load_dotenv
from util.train import train_model
from util.sahkotin import update_spot
from util.fingrid import update_nuclear
from util.entso_e import entso_e_nuclear
from util.fmi import update_wind_speed, update_temperature

"""
This script predicts the spot price and runs daily at 7 am (Finnish time). 
It enables the evaluation of a model available at the following link: 
[GitHub - vividfog/nordpool-predict-fi](https://github.com/vividfog/nordpool-predict-fi).

The prediction for the current day remains unchanged daily, while the model is retrained everyday,
facilitating the comparison of model performance.
"""


if __name__ == "__main__":
    try:
        load_dotenv('.env.local')  # take environment variables from .env.local
    except Exception as e:
        print(f"Error loading .env.local file. Did you create one? See README.md.")

    # Fetch mandatory environment variables and raise exceptions if they are missing
    def get_mandatory_env_variable(name):
        value = os.getenv(name)
        if value is None:
            raise ValueError(f"Mandatory variable {name} not set in environment")
        return value

    try:
        # Configuration and secrets, mandatory:
        fingrid_api_key = get_mandatory_env_variable('FINGRID_API_KEY')
        entso_e_api_key = get_mandatory_env_variable('ENTSO_E_API_KEY')
        fmisid_ws_env = get_mandatory_env_variable('FMISID_WS')
        fmisid_t_env = get_mandatory_env_variable('FMISID_T')
        fmisid_ws = ['ws_' + id for id in fmisid_ws_env.split(',')]
        fmisid_t = ['t_' + id for id in fmisid_t_env.split(',')]

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    df = pd.read_csv('data/data.csv')
    tmp = df.copy()

    # Ensure 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
        
    # We operate from this moment back and forward
    now = pd.Timestamp.utcnow()

    # Round up to the next full hour if not already on a full hour
    if now.minute > 0 or now.second > 0 or now.microsecond > 0:
        now = now.ceil('h')  # Rounds up to the nearest hour
        
    # Drop rows that are older than a week, unless we intend to do a retrospective prediction update after a model update
    #df = df[df.index > now - pd.Timedelta(days=7)]

    # Forward-fill the timestamp column for 5*24 = 120 hours ahead
    start_time = now + pd.Timedelta(hours=1)  # Start from the next hour
    end_time = now + pd.Timedelta(hours=120)  # 5 days ahead
    new_index = pd.date_range(start=start_time, end=end_time, freq='h')
    df = df.reindex(df.index.union(new_index))

    # Reset the index to turn 'timestamp' back into a column before the update functions
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Timestamp'}, inplace=True)

    # Get the latest FMI wind speed values for the data frame, past and future
    # NOTE: To save on API calls, this won't backfill history beyond 7 days even if asked
    df = update_wind_speed(df)
            
    # Get the latest FMI temperature values for the data frame, past and future
    # NOTE: To save on API calls, this won't backfill history beyond 7 days even if asked
    df = update_temperature(df)

    # Get the latest nuclear power data for the data frame, and infer the future from last known value
    # NOTE: To save on API calls, this won't backfill history beyond 7 days even if asked
    df = update_nuclear(df, fingrid_api_key=fingrid_api_key)

    # Fetch future nuclear downtime information from ENTSO-E unavailability data, h/t github:@pkautio
    df_entso_e = entso_e_nuclear(entso_e_api_key)

    # Refresh the previously inferred nuclear power numbers with the ENTSO-E data
    for index, row in df_entso_e.iterrows():
        mask = (df['Timestamp'] == row['timestamp'])
        df.loc[mask, 'NuclearPowerMW'] = row['NuclearPowerMW']

    for index, row in df.iterrows():
        if np.isnan(row['NuclearPowerMW']):
            df.loc[index, 'NuclearPowerMW'] = tmp.loc[index, 'NuclearPowerMW']

    # Get the latest spot prices for the data frame, past and future if any
    # NOTE: To save on API calls, this won't backfill history beyond 7 days even if asked
    df = update_spot(df)

    # TODO: Decide if including wind power capacity is necessary; it seems to worsen the MSE and R2
    # For now we'll drop it
    #df = df.drop(columns=['WindPowerCapacityMW'])

    # print("Filled-in dataframe before predict:\n", df)
    print("→ Days of data coverage (should be 7 back, 5 forward for now): ", int(len(df)/24))

    # Fill in the 'hour', 'day_of_week', and 'month' columns for the model
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['day_of_week'] = df['Timestamp'].dt.dayofweek + 1
    df['hour'] = df['Timestamp'].dt.hour
    df['month'] = df['Timestamp'].dt.month

    df = df.rename(columns={'Timestamp': 'timestamp'})

    specific_timestamp = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) 
    # Get the current time in Helsinki timezone
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    now = datetime.now(tz=helsinki_tz)

    # Convert to UTC+0 timezone
    utc_tz = pytz.timezone('UTC')
    specific_timestamp = specific_timestamp.astimezone(utc_tz)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    row = df[df['timestamp'] == specific_timestamp]

    df_nan = df.dropna()

    mae, mse, r2, samples_mae, samples_mse, samples_r2, rf_trained = train_model(df_nan, fmisid_ws=fmisid_ws, fmisid_t=fmisid_t)

    price_df = rf_trained.predict(df[['day_of_week', 'hour', 'month', 'NuclearPowerMW'] + fmisid_ws + fmisid_t])
    if 'PricePredict_cpkWh' in df.columns:
        df = df.drop(columns=['PricePredict_cpkWh'])
    for i in range(row.index[0]+1, len(price_df)):
        df.loc[i, 'predicted_spot_price'] = price_df[i]

    df.to_csv('data/data.csv', index=False)