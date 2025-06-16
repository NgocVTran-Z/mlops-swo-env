
import pandas as pd
import numpy as np
import sys
import gc

import os
sys.path.append(os.path.abspath(".."))


import s3fs
from typing import List

from common import *
from params import *
from transform import transform, tracking_transforming_input
from intervals import get_interval_from_transformed

from models import *
from visualize import *

from pyarrow.dataset import field

import sagemaker
from sagemaker import get_execution_role
import matplotlib.pyplot as plt
import seaborn as sns

# get the lastest saved data from mlflow run
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path





def get_training_path(motor, date_folders, tag_name, tag_name_digital):
    # get the list of filepath
    run_names_grp = []
    # for motor in motors:
    

    run_names = [f"Filtering_{tag_name}_by_{tag_name_digital}_{date_folder}" for date_folder in date_folders]
    run_names_grp.append(run_names)
    run_names_grp = list(np.concatenate(run_names_grp))

    # 
    training_paths = [get_param(
        experiment_name=experiment_name_FilteringByDigitalInput,
        experiment_run=run_name,
        keyword="saved_filtered_speed_filepath"
    ) for run_name in run_names_grp]
    
    return training_paths



def read_df(lst_training_paths):
    columns = [
        # "tag_name", 
        "value", 
        "time_utc", 
        # "units"
    ]
    dfs = []
    for path in lst_training_paths:
        try:
            # Chỉ đọc 2 cột A và B
            df = pd.read_parquet(path, columns=columns)
            dfs.append(df)
    
        except Exception as e:
            print(f"❌ error reading file {path}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        return "no data to be saved"



def prepare_training_data_clustering(
    motor,
    date_folders,
    tag_name, 
    tag_name_digital
):
    # get all data training path (from different folder i.e. 2025-03 2024-11 etc.
    lst_training_paths = get_training_path(motor=motor, date_folders=date_folders, tag_name=tag_name, tag_name_digital=tag_name_digital)
    combined_df = read_df(lst_training_paths=lst_training_paths) # read in a df
    
    # save df to b3 bucket
    destination_parquet_folder = f"s3://s3-assetcare-bucket/features_store/cluster_datasets/"
    
    current_time = get_current_timestamp_string()
    filename = f"{tag_name}_{current_time}.parquet"
    
    save_df_to_s3_parquet(
        df=combined_df,
        s3_path=destination_parquet_folder,
        filename=filename
    )
    gc.collect()

    
    return combined_df, lst_training_paths, destination_parquet_folder, filename





