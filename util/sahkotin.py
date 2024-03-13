import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import urllib.parse
import sys
from dotenv import load_dotenv
import os
import json


def get_url_nordpool(type, variable, market_region_2_char, start_time, end_time):
    if type == "dayahead":
        url = "https://marketdata-api.nordpoolgroup.com/"+type+"/"
        if variable == "price":
            url += "prices/area?deliveryA"
            url += "rea=" + str(market_region_2_char) + "&currency=EUR"
        elif variable == "volume-purchase":
            url += "volumes/area/purchases?deliverya"
            url += "rea=" + str(market_region_2_char)
        elif variable == "volume-sale":
            url += "volumes/area/sales?deliverya"
            url += "rea=" + str(market_region_2_char) 
    elif type == "regulationmarket":
        url = "https://marketdata-api.nordpoolgroup.com/"+type+"/"
        if variable == "price_up":
            url = url + "/v2/prices/up?deliveryarea="+str(market_region_2_char)+"&currency=EUR"
        elif variable == "price_down":
            url = url + "/v2/prices/down?deliveryarea="+str(market_region_2_char)+"&currency=EUR"
        elif variable == "volume_ordinary_up":
            url = url + "/v2/volumes/ordinary/up?deliveryarea="+str(market_region_2_char)
        elif variable == "volume_ordinary_down":
            url = url + "/v2/volumes/ordinary/down?deliveryarea="+str(market_region_2_char)
        elif variable == "volume_automatic_up":
            url = url + "/v2/volumes/automatic/up?deliveryarea="+str(market_region_2_char)
        elif variable == "volume_automatic_down":
            url = url + "/v2/volumes/automatic/down?deliveryarea="+str(market_region_2_char)
    url+="&startTime=" + urllib.parse.quote(start_time) +"&endTime=" + urllib.parse.quote(end_time)
    return url

def get_mandatory_env_variable(name):
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Mandatory variable {name} not set in environment")
    return value

def fetch_data_nordpool(url_data):
    try:
        load_dotenv('.env.local')  # take environment variables from .env.local
    except Exception as e:
        print(f"Error loading .env.local file. Did you create one? See README.md.")

    NORDPOOL_TOKEN = get_mandatory_env_variable('NORDPOOL_TOKEN')
    PAYLOAD_NORDPOOL = get_mandatory_env_variable('PAYLOAD_NORDPOOL')

    try:
        url = "https://sts.nordpoolgroup.com/connect/token"

        headers = {
        'Authorization': 'Basic '+str(NORDPOOL_TOKEN)+'',
        'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.request("POST", url, headers=headers, data=PAYLOAD_NORDPOOL)
        token = json.loads(response.text)['access_token']
        if token != None:
            headers = {
                'Authorization': 'Bearer '+ str(token),
            }
        
        response = requests.request("GET", url_data, headers=headers, data="")
        if response.status_code != 200:
            if response.status_code == 204:
                raise (f"Status code: {response.status_code}")
            return None
        else:
            data = json.loads(response.text)
            trasformed_data = []
            for item in data:
                unit = item['unit']
                scale = item['scale']
                values = item['values']
                for value in values:
                    trasformed_data.append((value['startTime'], value['endTime'], value['value']))
            return trasformed_data
    except requests.exceptions.HTTPError as errh:
        raise (f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        raise (f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        raise (f"Timeout Error: {errt}")


def fetch_electricity_price_data(start_date, end_date):
    """
    Fetches electricity price data from nordpool for a specified date range.

    Parameters:
    - start_date (str): The start datetime in ISO format.
    - end_date (str): The end datetime in ISO format.

    Returns:
    - pd.DataFrame: A DataFrame with two columns ['Timestamp', 'true_spot_price'] where 'Timestamp' is the datetime and 'true_spot_price' is the electricity price.
    """
    data_array = {}
    url_price = get_url_nordpool("dayahead", "price", 'FI', start_date, end_date)
    print(url_price)
    data_array['price'] = fetch_data_nordpool(url_price)
    print(data_array)

    df = pd.DataFrame(data_array['price'], columns=['start', 'end', 'value'])
    if not df.empty:
        try:
            df['start'] = pd.to_datetime(df['start'], utc=True)
            df = df.drop(columns=['end'])
            df['value'] = pd.to_numeric(df['value'])
            df = df.rename(columns={'start': 'Timestamp', 'value': 'true_spot_price'})
            df = df.sort_values(by='Timestamp')
        except Exception as e:
            print(f"Error processing data from Sähkötin API: {e}")
            sys.exit(1)
        return df
    else:
        print("No data returned from the API.")
        return pd.DataFrame(columns=['Timestamp', 'true_spot_price'])



def clean_up_df_after_merge(df):
    """
    This function removes duplicate columns resulting from a merge operation,
    and fills the NaN values in the original columns with the values from the
    duplicated columns. Assumes duplicated columns have suffixes '_x' and '_y',
    with '_y' being the most recent values to retain.
    """
    # Identify duplicated columns by their suffixes
    cols_to_remove = []
    for col in df.columns:
        if col.endswith('_x'):
            original_col = col[:-2]  # Remove the suffix to get the original column name
            duplicate_col = original_col + '_y'
            
            # Check if the duplicate column exists
            if duplicate_col in df.columns:
                # Fill NaN values in the original column with values from the duplicate
                df[original_col] = df[col].fillna(df[duplicate_col])
                
                # Mark the duplicate column for removal
                cols_to_remove.append(duplicate_col)
                
            # Also mark the original '_x' column for removal as it's now redundant
            cols_to_remove.append(col)
    
    # Drop the marked columns
    df.drop(columns=cols_to_remove, inplace=True)
    
    return df

def update_spot(df):
    """
    Updates the input DataFrame with electricity price data fetched from sahkotin.fi.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with electricity price data.
    """
    current_date = datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    history_date = (datetime.now(pytz.UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_date = (datetime.now(pytz.UTC) + timedelta(hours=120)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    print(f"* Fetching electricity price data between {history_date[:10]} and {end_date[:10]}")
    
    price_df = fetch_electricity_price_data(history_date, end_date)   
    if not price_df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
        merged_df = pd.merge(df, price_df, on='Timestamp', how='left')
       
        merged_df = clean_up_df_after_merge(merged_df)
               
        return merged_df
    else:
        print("Warning: No electricity price data fetched; unable to update DataFrame.")
        return df
