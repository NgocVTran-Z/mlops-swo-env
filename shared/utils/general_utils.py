import boto3
import pandas as pd
from io import BytesIO

def load_parquet_from_s3(s3_client, bucket, key):
    """
    Load a .parquet file from S3 into a pandas DataFrame.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))
