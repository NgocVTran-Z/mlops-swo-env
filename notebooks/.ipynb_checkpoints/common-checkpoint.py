import boto3
import pandas as pd
from io import StringIO

import s3fs
from typing import List
import pyarrow.dataset as ds
from pyarrow.dataset import field

import mlflow
from mlflow.tracking import MlflowClient
import gc
from datetime import datetime


def get_current_timestamp_string() -> str:
    """
    Trả về timestamp hiện tại dạng 'YYYYMMDD_HHMMSS', ví dụ: '20250401_134502'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")



def save_df_to_s3_parquet(
    df: pd.DataFrame, 
    s3_path: str, 
    filename: str = "output.parquet") -> None:
    """
    Lưu DataFrame dạng parquet lên S3.

    Args:
        df (pd.DataFrame): Dữ liệu cần lưu
        s3_path (str): Đường dẫn folder S3 (ví dụ: 's3://bucket/folder/')
        filename (str): Tên file lưu (mặc định là 'output.parquet')
    """
    if not s3_path.endswith("/"):
        s3_path += "/"
        
    full_path = s3_path + filename

    # Dùng s3fs để ghi parquet trực tiếp lên S3
    fs = s3fs.S3FileSystem()
    with fs.open(full_path, 'wb') as f:
        df.to_parquet(f, index=False)
    
    print(f"✅ Saved to {full_path}")


def read_parquet_from_s3(bucket_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Read all .parquet files in a folder on S3, selecting only specific columns.

    Args:
        bucket_path (str): S3 path, e.g., 's3://my-bucket/my-folder/'
        columns (List[str]): List of columns to read

    Returns:
        pd.DataFrame: DataFrame containing merged data from all parquet files
    """
    fs = s3fs.S3FileSystem(anon=False)
    file_list = fs.glob(f"{bucket_path.rstrip('/')}/*.parquet")

    if not file_list:
        raise FileNotFoundError(f"No .parquet files found in {bucket_path}")

    dfs = []
    for file in file_list:
        try:
            df = pd.read_parquet(file, columns=columns, filesystem=fs)
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No valid parquet files were read.")



# def load_csv_files(filepaths, usecols=None, dtype=None):
#     """
#     Efficiently load multiple CSV files into a single DataFrame.

#     Args:
#         filepaths (List[str]): List of CSV file paths.
#         usecols (List[str], optional): List of columns to load. Defaults to None (load all).
#         dtype (dict, optional): Dictionary of column types to optimize memory. Defaults to None.

#     Returns:
#         pd.DataFrame: Concatenated DataFrame of all CSVs.
#     """
#     dfs = []
#     for path in filepaths:
#         try:
#             df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=True, header=None)
#             dfs.append(df)
#         except Exception as e:
#             print(f"Error reading {path}: {e}")

#     return pd.concat(dfs, ignore_index=True)



def read_parquet_with_filter(bucket_path: str, columns: List[str], filter_expr=None) -> pd.DataFrame:
    """
    Đọc Parquet từ S3 với pyarrow + filter, chỉ chọn 1 số column.

    Args:
        bucket_path (str): Dạng 's3://bucket/path/'
        columns (List[str])
        filter_expr: pyarrow.dataset expression

    Returns:
        pd.DataFrame
    """
    # Parse bucket + key
    if bucket_path.startswith("s3://"):
        bucket_path = bucket_path.replace("s3://", "")  # bỏ scheme
    s3 = s3fs.S3FileSystem(anon=False)

    dataset = ds.dataset(
        source=bucket_path,
        format="parquet",
        filesystem=s3
    )

    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pandas()


#------------mlflow support functions--------
def get_param(
    experiment_name: str,
    experiment_run: str,
    keyword: str,
    run_id=False
):
    try:
        # get the lastest saved data from mlflow run
        client = MlflowClient()
        
        # Lấy ID của experiment (ví dụ từ tên)
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        # print("exp id:", experiment_id)
    
        filter_string = f'tags.mlflow.runName = "{experiment_run}"'
        # print(filter_string)
        
        # Lấy danh sách run thuộc experiment đó
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=1
        )
        # print(runs)
        
        # Lấy latest run
        if runs:
            latest_run = runs[0]
            run_id = latest_run.info.run_id
            params = latest_run.data.params
            # print("Latest run_id:", run_id)
            # print("Params:", params)
            return params[keyword]
        else:
            print("No matching run found.")
    
        # print(params)
    except Exception as e:
        print(experiment_name, "with keyword:", keyword)
        pass




#======================================== new management

def get_child_run_param_by_name(
    experiment_name: str,
    parent_run_name: str,
    child_keyword: str,
    param_key: str
):
    try:
        client = MlflowClient()

        # Get experiment ID
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return None
        experiment_id = experiment.experiment_id

        # Find parent run by name
        parent_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags.mlflow.runName = "{parent_run_name}"',
            order_by=["start_time DESC"],
            max_results=1
        )

        if not parent_runs:
            print(f"No parent run with name '{parent_run_name}' found.")
            return None

        parent_run = parent_runs[0]
        parent_run_id = parent_run.info.run_id

        # Find child runs of this parent
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
            order_by=["start_time DESC"],
        )

        # Search for a child run that contains the keyword in the name
        for run in child_runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if child_keyword in run_name:
                return run.data.params.get(param_key, f"Param '{param_key}' not found.")

        print(f"No child run matching keyword '{child_keyword}' found.")
        return None

    except Exception as e:
        print("Error:", e)
        return None




def list_files_in_s3_folder(bucket: str, prefix: str, end_with_csv: bool = False):
    """
    List all level-1 files directly under a given S3 folder prefix.
    Optionally filters to only include .csv files.

    Parameters:
    - bucket (str): Name of the S3 bucket (without 's3://').
    - prefix (str): Folder prefix path, must end with '/'.
    - end_with_csv (bool): If True, only include files ending with '.csv'.

    Returns:
    - List[str]: List of object keys representing files directly under the given prefix.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_path = key[len(prefix):]

            # Include only level-1 files (no subfolders)
            if "/" not in relative_path:
                # If end_with_csv is True, only include .csv files
                if not end_with_csv or key.endswith(".csv"):
                    files.append(key)

    return files


# save to ...
def save_dataframe_to_s3_in_batches(
    df: pd.DataFrame, 
    s3_path_prefix: str, 
    header,
    batch_size: int = 100000,
):
    """
    Save a DataFrame to S3 in multiple CSV files, each with a batch of rows.

    Parameters:
    - df: DataFrame to save
    - s3_path_prefix: Full S3 path prefix (e.g. "s3://bucket/folder1/folder2/")
    - batch_size: Number of rows per file
    """
    import tempfile
    import s3fs

    # Tách phần bucket và prefix từ s3_path_prefix
    if not s3_path_prefix.endswith('/'):
        s3_path_prefix += '/'
    
    # Dùng s3fs để save file trực tiếp lên S3
    fs = s3fs.S3FileSystem()

    total_rows = len(df)
    total_batches = (total_rows + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch_df = df[i * batch_size: (i + 1) * batch_size]
        file_name = f"part_{i:04d}.csv"
        full_s3_path = os.path.join(s3_path_prefix, file_name)

        # Save lên S3
        with fs.open(full_s3_path, 'w') as f:
            batch_df.to_csv(f, index=False, header=header)
        print(f"✅ Saved batch {i+1}/{total_batches} to {full_s3_path}")




def load_csv_files(filepaths, usecols=None, dtype=None):
    """
    Efficiently load multiple CSV files into a single DataFrame.

    Args:
        filepaths (List[str]): List of CSV file paths.
        usecols (List[str], optional): List of columns to load. Defaults to None (load all).
        dtype (dict, optional): Dictionary of column types to optimize memory. Defaults to None.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all CSVs.
    """
    dfs = []
    for path in filepaths:
        try:
            df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=True, header=None)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return pd.concat(dfs, ignore_index=True)



def get_value(run_id, keys):
    run = mlflow.get_run(run_id)

    # Lấy toàn bộ params
    params = run.data.params
    # print(params)
    
    # Lấy giá trị cụ thể, ví dụ: cluster_id
    filepath = [params.get(key) for key in keys]
    return filepath







