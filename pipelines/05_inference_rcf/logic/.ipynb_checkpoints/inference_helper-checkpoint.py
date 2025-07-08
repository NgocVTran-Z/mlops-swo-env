import boto3
import numpy as np
import io
import pandas as pd
from urllib.parse import urlparse

def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def predict_batch_rcf(df, endpoint_name):
    runtime = boto3.client('sagemaker-runtime')

    # Convert dataframe to CSV string format
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)
    payload = csv_buffer.getvalue()

    # Call endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )

    result = json.loads(response['Body'].read().decode('utf-8'))
    scores = result['scores']

    return scores


def transform(df):
    # Convert data types and sort
    df['value'] = df['value'].astype('float32')
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df = df.sort_values(by='time_utc').reset_index(drop=True)

    # Check if DataFrame is empty
    if df.empty:
        mlflow.log_param("data_status", "empty")
        mlflow.log_metric("rows_processed", 0)
        print("⚠️ DataFrame is empty. Skipping transform.")
        return df  # hoặc return df nếu bạn muốn vẫn trả về một DataFrame rỗng
        
    
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