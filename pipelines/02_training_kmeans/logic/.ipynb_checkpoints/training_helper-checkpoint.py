import pandas as pd
import boto3
import joblib
from sklearn.cluster import KMeans
from shared.utils.general_utils import read_parquet_from_s3

def save_model_to_s3(model, bucket, model_key):
    joblib.dump(model, "/tmp/model.joblib")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/model.joblib", bucket, model_key)

def save_dataframe_to_s3(df, bucket, output_key):
    out_buffer = df.to_parquet(index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=output_key, Body=out_buffer)

def run_training(bucket, input_key, model_output_key, clustered_output_key, n_clusters):
    print("Loading data from S3...")
    df = read_parquet_from_s3(bucket, input_key)

    print("Training KMeans model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df[["value"]])  # 🚩 không scale dữ liệu, train trực tiếp

    print("Saving model to S3...")
    save_model_to_s3(kmeans, bucket, model_output_key)

    print("Generating cluster labels...")
    df["cluster"] = kmeans.predict(df[["value"]])

    print("Saving clustered data to S3...")
    save_dataframe_to_s3(df, bucket, clustered_output_key)

    print("Training pipeline completed.")
