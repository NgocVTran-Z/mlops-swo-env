import pandas as pd
import sys
import os

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



def internal_preprocessing(df, filename, tag):
    print("🧠 This is internal logic of preprocessing pipeline", tag)
    # df["value"] = df["value"] * 200
    # print(f"🔍 Preview of {filename}:\n", df.head())

    # get mapping tag name from the input tag
    digital_tag = mapping_tags["Digital"][tag]
    speed_tag = mapping_tags["Speed"][tag]
    # cloudwatch log debug
    print("digital tag", digital_tag)
    print("speed tag", speed_tag)

    
    df_digital = df[df["tag_name"]==digital_tag]
    df_speed = df[df["tag_name"]==speed_tag]

    # transform raw df to df with every-single-second datapoints
    df_digital_ = transform(df_digital)
    df_speed_ = transform(df_speed)
    print("Transform: df_digital", df_digital_.shape)
    print("Transform: df_speed", df_speed_.shape)
    
    # get the interval of digital df
    df_digital_interval = get_interval_from_transformed(df_digital_)
    
    # filtered speed
    filtered_speed = filter_by_intervals(df_digital_interval, df_speed_)

    # cloudwatch log debug
    print("digital tag", digital_tag)
    print("speed tag", speed_tag)
    print("Transform: df_digital", df_digital_.shape)
    print("Transform: df_speed", df_speed_.shape)
    print("Invertal digital", df_digital_interval.head())
    print("Filtered speed", filtered_speed.shape)
    
    return df



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

