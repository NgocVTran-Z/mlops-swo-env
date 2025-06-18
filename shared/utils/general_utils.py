import boto3
import pandas as pd
from io import BytesIO






def load_parquet_from_s3(s3_client, bucket, key):
    """
    Load a .parquet file from S3 into a pandas DataFrame.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))








def get_current_timestamp_string() -> str:
    """
    Return current timestamp 'YYYYMMDD_HHMMSS'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def transform(df):
    """
    Transform the raw input df into df with datapoint in every single second.
    Works directly on the original df (in-place safe).
    """
    import pandas as pd
    import mlflow

    # Convert data types and sort
    df.loc[:, 'value'] = df['value'].astype('float32')
    df.loc[:, 'time_utc'] = pd.to_datetime(df['time_utc'])
    df.sort_values(by='time_utc', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Check if DataFrame is empty
    if df.empty:
        mlflow.log_param("data_status", "empty")
        mlflow.log_metric("rows_processed", 0)
        print("⚠️ DataFrame is empty. Skipping transform.")
        return df

    # Handle the case where the first value is 0
    if df.iloc[0]['value'] == 0:
        non_zero_idx = df[df['value'] != 0].index
        if not non_zero_idx.empty:
            first_non_zero_idx = non_zero_idx[0]
            replacement_value = df.at[first_non_zero_idx, 'value']
            df.loc[:first_non_zero_idx - 1, 'value'] = replacement_value

    # Expand time series with 1-second intervals
    expanded_data = []
    for i in range(len(df) - 1):
        if i % 10000 == 0:
            print(f"Processing row {i}")
        start_time = df.at[i, 'time_utc']
        end_time = df.at[i + 1, 'time_utc'] - pd.Timedelta(seconds=1)
        time_range = pd.date_range(start=start_time, end=end_time, freq='S')

        row = df.iloc[i].copy()
        row_dict = row.to_dict()
        expanded_data.extend([{**row_dict, 'time_utc': t} for t in time_range])

    # Append the last row
    expanded_data.append(df.iloc[-1].to_dict())

    df_expanded = pd.DataFrame(expanded_data)
    
    return df_expanded




def get_interval_from_transformed(df):
    """
    detect time interval which has >= 0 value of digital tag
    """
    intervals = []
    start_time = None
    values = []
    
    for i in range(len(df)):
        if df.iloc[i]["value"] != 0:
            if start_time is None:
                start_time = df.iloc[i]['time_utc']
                value = df.iloc[i]["value"].item()
        else:
            if start_time is not None:
                intervals.append({'from': start_time, 
                                  'to': df.iloc[i]['time_utc'], 
                                  "value": value})
                start_time = None

    if start_time is not None:
        print("start time has some values")

    df_intervals = pd.DataFrame(intervals)
    return df_intervals




