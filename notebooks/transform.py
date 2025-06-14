import pandas as pd
import mlflow
from utils.common import *
from config.params import *
from preprocessing.intervals import get_interval_from_transformed, filter_by_intervals

import matplotlib.pyplot as plt
import gc

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



def tracking_transform_analog(
    source_parquet_folder: str,
    tag_name: str,
    destination_parquet_folder: str,
    date_folder: str,
    current_time: str,
):

    # current_time = str(get_current_timestamp_string())
    mlflow.set_experiment(experiment_name_AnalogTransformation)

    with mlflow.start_run(run_name=f"Transforming_{tag_name}_{date_folder}") as run:
        # 1. get df by digital tag name
        df_input = read_parquet_with_filter(
            bucket_path=source_parquet_folder,
            columns=columns,
            filter_expr=(field("tag_name") == tag_name)
        )
        # log df info
        mlflow.log_param("source_parquet_folder", source_parquet_folder)
        mlflow.log_param("tag_name", tag_name)
        mlflow.log_param("original_shape", df_input.shape)
        mlflow.log_param("columns", df_input.columns)
    
        # 2. transform data
        transformed_analog = transform(df=df_input)
        del df_input
        gc.collect()
        
        # save and log transformed data
        _ = save_df_to_s3_parquet(
            df=transformed_analog,
            s3_path=destination_parquet_folder,
            filename=f"{tag_name}_{current_time}.parquet"
        )
        # log info
        mlflow.log_param(f"{tag_name} transformed saved path", destination_parquet_folder)
        # Log shape 
        mlflow.log_param(f"{tag_name}_num_rows", transformed_analog.shape[0])
        mlflow.log_param(f"{tag_name}_num_cols", transformed_analog.shape[1])
        mlflow.log_param("destination_parquet_file", destination_parquet_folder + f"{tag_name}_{current_time}.parquet")
        
        del transformed_analog
        gc.collect()
        del _
        gc.collect()
        
    return run.info.run_id




# -------------- mlflow function
def tracking_transforming_input(
    source_parquet_folder,
    tag_name,
    destination_parquet_folder,
    intervals_folder,
    date_folder,
    current_time,
    input_type="digital"
):
    # current_time = str(get_current_timestamp_string())
    if input_type=="digital":
        mlflow.set_experiment(experiment_name_DigitalTransformation)
    elif input_type=="analog":
        mlflow.set_experiment(experiment_name_AnalogTransformation)
        
    with mlflow.start_run(run_name=f"Transforming_{tag_name}_{date_folder}") as run:
        # 1. get df by digital tag name
        df_input = read_parquet_with_filter(
            bucket_path=source_parquet_folder,
            columns=columns,
            filter_expr=(field("tag_name") == tag_name)
        )
        # log df info
        mlflow.log_param("source_parquet_folder", source_parquet_folder)
        mlflow.log_param("tag_name", tag_name)
        mlflow.log_param("original_shape", df_input.shape)
        mlflow.log_param("columns", df_input.columns)
    
        # 2. transform data
        transformed_digital = transform(df=df_input)
        del df_input
        # save and log transformed data
        save_df_to_s3_parquet(
            df=transformed_digital,
            s3_path=destination_parquet_folder,
            filename=f"{tag_name}_{current_time}.parquet"
        )
        # log info
        mlflow.log_param(f"{tag_name} transformed saved path", destination_parquet_folder)
        # Log shape 
        mlflow.log_param(f"{tag_name}_num_rows", transformed_digital.shape[0])
        mlflow.log_param(f"{tag_name}_num_cols", transformed_digital.shape[1])
        mlflow.log_param("destination_parquet_file", destination_parquet_folder + f"{tag_name}_{current_time}.parquet")
    
        # 3. get interval data
        df_intervals = get_interval_from_transformed(transformed_digital)
        del transformed_digital
        # Save digital intervals
        save_df_to_s3_parquet(
            df=df_intervals,
            s3_path=intervals_folder,
            filename=f"{tag_name}_{current_time}.parquet"
        )
        # Log shape 
        mlflow.log_param(f"{tag_name}_interval_num_rows", df_intervals.shape[0])
        mlflow.log_param(f"{tag_name}_interval_num_cols", df_intervals.shape[1])
        # ✅ Log interval file path
        mlflow.log_param("interval_parquet_file", intervals_folder + f"{tag_name}_{current_time}.parquet")
        
        print("✅ Logged digital tag:", tag_name)
        print("*********")
        return run.info.run_id



#============================================= new management


def tracking_transforming_input2(
    source_parquet_folder,
    tag_name,
    destination_parquet_folder,
    intervals_folder,
    date_folder,
    current_time,
    experiment_name,
    parent_run_id
    # input_type="digital",
):
    current_time = str(get_current_timestamp_string())
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name=f"Transforming_{tag_name}_{date_folder}", nested=True) as run:
            # 1. get df by digital tag name
            df_input = read_parquet_with_filter(
                bucket_path=source_parquet_folder,
                columns=columns,
                filter_expr=(field("tag_name") == tag_name)
            )
            # log df info
            mlflow.log_param("source_parquet_folder", source_parquet_folder)
            mlflow.log_param("tag_name", tag_name)
            mlflow.log_param("original_shape", df_input.shape)
            mlflow.log_param("columns", df_input.columns)
        
            # 2. transform data
            transformed_digital = transform(df=df_input)
            del df_input
            gc.collect()
            # save and log transformed data
            save_df_to_s3_parquet(
                df=transformed_digital,
                s3_path=destination_parquet_folder,
                filename=f"{tag_name}_{current_time}.parquet"
            )
            # log info
            mlflow.log_param(f"{tag_name} transformed saved path", destination_parquet_folder)
            # Log shape 
            mlflow.log_param(f"{tag_name}_num_rows", transformed_digital.shape[0])
            mlflow.log_param(f"{tag_name}_num_cols", transformed_digital.shape[1])
            mlflow.log_param("destination_parquet_file", destination_parquet_folder + f"{tag_name}_{current_time}.parquet")
        
            # 3. get interval data
            df_intervals = get_interval_from_transformed(transformed_digital)
            del transformed_digital
            gc.collect()
            # Save digital intervals
            save_df_to_s3_parquet(
                df=df_intervals,
                s3_path=intervals_folder,
                filename=f"{tag_name}_{current_time}.parquet"
            )
            # Log shape 
            mlflow.log_param(f"{tag_name}_interval_num_rows", df_intervals.shape[0])
            mlflow.log_param(f"{tag_name}_interval_num_cols", df_intervals.shape[1])
            # ✅ Log interval file path
            mlflow.log_param("interval_parquet_file", intervals_folder + f"{tag_name}_{current_time}.parquet")
            
            print("✅ Logged digital tag:", tag_name)
            print("*********")
            del df_intervals
            gc.collect()
            return run.info.run_id



def filtering_speed_by_digital(
    motor: str,
    date_folder: str,
    current_time: str
):
    mlflow.set_experiment(experiment_name_FilteringByDigitalInput)

    # tag_name_speed = f"{motor}_SPEED_REF"
    tag_name_speed = f"{motor}_ACTUAL_MOTOR_SPEED"
    tag_name_interval = f"{motor}_INVERTER_RUNNING"
    # print(tag_name_speed)

    experiment_run_speed = f"Transforming_{tag_name_speed}_{date_folder}"
    experiment_run_interval = f"Transforming_{tag_name_interval}_{date_folder}"
    # print(experiment_run_speed)

    # 👇 Đảm bảo không có run nào đang active
    if mlflow.active_run():
        mlflow.end_run()

    
    run_name = f"Filtering_{tag_name_speed}_by_{tag_name_interval}_{date_folder}"
    with mlflow.start_run(run_name=run_name):

        # get SPEED transformed data
        transformed_speed_path = get_child_run_param_by_name(
            experiment_name=experiment_name_RegularInterval,
            parent_run_name=experiment_name_RegularInterval_SpeedTagNames,
            child_keyword=tag_name_speed,
            param_key="destination_parquet_file"
        )
        
        print("This->", transformed_speed_path)
        transformed_speed = pd.read_parquet(transformed_speed_path)
        # mlflow.log_param("transformed_speed_shape", transformed_speed.shape)
    
        # get INTERVAL data
        interval_path = get_child_run_param_by_name(
            experiment_name=experiment_name_RegularInterval,
            parent_run_name=experiment_name_RegularInterval_DigitalInput,
            child_keyword=tag_name_interval,
            param_key="interval_parquet_file"
        )
        

        
        df_intervals = pd.read_parquet(interval_path)
        if df_intervals.shape == (0,0):
            print("no interval")
            return 
            
        # get filtered speed
        filtered_speed = filter_by_intervals(df_intervals, transformed_speed)
        mlflow.log_metric("filtered_rows", filtered_speed.shape[0])
        print(filtered_speed.shape)
        del transformed_speed, df_intervals
        gc.collect()
    
        # Step 5: Remove outliers
        before_outlier = filtered_speed.shape[0]
        filtered_speed = filtered_speed[(filtered_speed["value"] >= -3000) & (filtered_speed["value"] <= 3000)]
        after_outlier = filtered_speed.shape[0]
        mlflow.log_metric("removed_outliers", before_outlier - after_outlier)
    
        # step 6: save 
        destination_parquet_folder = f"s3://s3-assetcare-bucket/features_store/filtered/{date_folder}/"
        filename = f"{tag_name_speed}_{current_time}.parquet"
        save_df_to_s3_parquet(
            df=filtered_speed,
            s3_path=destination_parquet_folder,
            filename=f"{tag_name_speed}_{current_time}.parquet"
        )
        mlflow.log_param("saved_filtered_speed_filepath", destination_parquet_folder + filename)
    
        # Step 7: Plot + log chart
        plt.figure(figsize=(12, 6))
        plt.scatter(filtered_speed["time_utc"], filtered_speed["value"], 
                    color="blue", alpha=0.7, label="Values")
        plt.xlabel("Datetime")
        plt.ylabel("Values")
        plt.title(f"Scatter Plot: {tag_name_speed}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Không cần savefig
        mlflow.log_figure(plt.gcf(), artifact_file=f"charts/scatter_{tag_name_speed}.png")
        plt.close()
        print(f"✅ Phase 2 tracking complete for {tag_name_speed}")

        del filtered_speed
        gc.collect()




