import pandas as pd
import sys
import os

import mlflow


# Add shared module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))

from utils.general_utils import transform


mapping_tags = {
    "Digital": {
        "DWA": "DWA_INVERTER_RUNNING",
        "DWB": "DWB_INVERTER_RUNNING",
        "DWC": "DWC_INVERTER_RUNNING",
        "TD": "TD_INVERTER_RUNNING",
    },
    "Speed": {
        "DWA": "DWA_ACTUAL_MOTOR_SPEED",
        "DWB": "DWB_ACTUAL_MOTOR_SPEED",
        "DWC": "DWC_ACTUAL_MOTOR_SPEED",
        "TD": "TD_ACTUAL_MOTOR_SPEED",
    }
   
}


#----params--
tracking_server_arn =  os.environ["TRACKING_SERVER_ARN"]
mlflow.set_tracking_uri(tracking_server_arn)
mlflow.set_experiment("01_preprocessing_kmeans")



def internal_preprocessing(
    df, 
    filename, 
    tag,
    output_filename, 
    output_path                                  
):
    print("This is internal logic of preprocessing pipeline", tag)

    # Get mapping tag names
    digital_tag = mapping_tags["Digital"][tag]
    speed_tag = mapping_tags["Speed"][tag]

    # Transform digital and speed data directly from original df (no intermediate slice)
    df_digital_ = transform(df.loc[df["tag_name"] == digital_tag].copy())
    df_speed_ = transform(df.loc[df["tag_name"] == speed_tag].copy())

    # Get intervals from transformed digital tag
    df_digital_interval = get_interval_from_transformed(df_digital_)

    # Filter speed data by intervals
    filtered_speed = filter_by_intervals(df_digital_interval, df_speed_)

    # Log for debugging
    print("digital tag", digital_tag)
    print("speed tag", speed_tag)
    print("Transform: df_digital", df_digital_.shape)
    print("Transform: df_speed", df_speed_.shape)
    print("Interval digital", df_digital_interval.shape)
    print("Filtered speed", filtered_speed.shape)
    # print("Filtered speed columns:", filtered_speed.columns)

    with mlflow.start_run(run_name=f"preprocessing_{tag}"):
        mlflow.set_tag("filename", filename)
        mlflow.set_tag("tag", tag)
        mlflow.log_param("digital_tag", digital_tag)
        mlflow.log_param("speed_tag", speed_tag)
        mlflow.log_metric("df_digital_rows", df_digital_.shape[0])
        mlflow.log_metric("df_speed_rows", df_speed_.shape[0])
        mlflow.log_metric("interval_count", df_digital_interval.shape[0])
        mlflow.log_metric("filtered_speed_rows", filtered_speed.shape[0])
        mlflow.log_metric("filtered_speed_columns", len(filtered_speed.columns))
        mlflow.log_metric("output_filename", output_filename) 
        mlflow.log_metric("output_path", output_path)
        
        mlflow.end_run()

    return filtered_speed




def get_interval_from_transformed(df):
    
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



def filter_by_intervals(df_intervals: pd.DataFrame, df_data: pd.DataFrame) -> pd.DataFrame:
    """
    return all row in df_data which timestamp (time_utc) i.
    """
    df_intervals = df_intervals[['from', 'to']].copy()
    df_data = df_data.copy()

    # Ensure datetime
    df_intervals['from'] = pd.to_datetime(df_intervals['from'])
    df_intervals['to'] = pd.to_datetime(df_intervals['to'])
    df_data['time_utc'] = pd.to_datetime(df_data['time_utc'])

    # Sorting
    df_intervals = df_intervals.sort_values('from')
    df_data = df_data.sort_values('time_utc')

    # tag timestamp in nearest
    merged = pd.merge_asof(
        df_data,
        df_intervals,
        left_on='time_utc',
        right_on='from',
        direction='backward'
    )

    # filter the row in right time frame
    filtered = merged[merged['time_utc'] <= merged['to']]

    # return original df columns
    return filtered[df_data.columns]

