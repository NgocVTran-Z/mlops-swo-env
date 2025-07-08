import os
import boto3
import pandas as pd
from io import BytesIO


def parse_s3_uri(s3_uri):
    """Convert s3://bucket/key → (bucket, key)"""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

def load_parquet_from_s3(s3_client, bucket, key):
    """Load a parquet file from S3 into a Pandas DataFrame"""
    print(f"📥 Loading file from S3 - Bucket: {bucket}, Key: {key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))

def main():
    file_path = os.environ.get("INPUT_S3_URI")
    tag_name = os.environ.get("TAG_NAME")
    endpoint_kmeans = os.environ.get("ENDPOINT_NAME_KMEANS")
    endpoint_rcf = os.environ.get("ENDPOINT_NAME_RCF")

    print("✅ Inference RCF script started")
    print(f"📂 INPUT_S3_URI: {file_path}")

    # Parse S3 URI to get bucket and key
    bucket, key = parse_s3_uri(file_path)

    # Init boto3 S3 client
    s3_client = boto3.client("s3")

    # Load the parquet file from S3
    df = load_parquet_from_s3(s3_client, bucket, key)

    # Filter by tag_name
    print(f"🏷️ TAG_NAME: {tag_name}")
    df = df[df["tag_name"] == tag_name].copy()
    print(f"📊 Data shape after filtering: {df.shape}")

    # Placeholder for next step
    print(f"🧠 KMeans Endpoint: {endpoint_kmeans}")
    print(f"🌲 RCF Endpoint: {endpoint_rcf}")

if __name__ == "__main__":
    main()
