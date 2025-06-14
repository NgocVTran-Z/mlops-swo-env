import pandas as pd


def get_interval_from_transformed(df):
    # Xác định các khoảng thời gian có giá trị khác 0
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
    Trả về các dòng trong df_data mà timestamp (time_utc) nằm trong bất kỳ khoảng thời gian nào của df_intervals.
    """
    df_intervals = df_intervals[['from', 'to']].copy()
    df_data = df_data.copy()

    # Ensure datetime
    df_intervals['from'] = pd.to_datetime(df_intervals['from'])
    df_intervals['to'] = pd.to_datetime(df_intervals['to'])
    df_data['time_utc'] = pd.to_datetime(df_data['time_utc'])

    # Sắp xếp
    df_intervals = df_intervals.sort_values('from')
    df_data = df_data.sort_values('time_utc')

    # Gán mỗi timestamp vào khoảng gần nhất trước đó
    merged = pd.merge_asof(
        df_data,
        df_intervals,
        left_on='time_utc',
        right_on='from',
        direction='backward'
    )

    # Lọc đúng các dòng thực sự nằm trong khoảng
    filtered = merged[merged['time_utc'] <= merged['to']]

    # Trả về đúng cột gốc của df_data (giờ không bị đổi tên nữa)
    return filtered[df_data.columns]

