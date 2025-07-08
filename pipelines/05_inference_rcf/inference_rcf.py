import os
import boto3
from io import BytesIO



def load_parquet_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))
    

def main():
    file_path = os.environ.get("INPUT_S3_URI")
    tag_name = os.environ.get("TAG_NAME")
    endpoint_kmeans = os.environ.get("ENDPOINT_NAME_KMEANS")
    endpoint_rcf = os.environ.get("ENDPOINT_NAME_RCF")

    data_prefix = os.environ["DATA_PREFIX"]
    
    print("✅ Inference RCF script started")
    
    print(f"✅ INPUT_S3_URI: {file_path} - load the raw data")
    # Init client
    s3_client = boto3.client("s3")

    # Load
    bucket = os.environ["S3_BUCKET"]
    df = load_parquet_from_s3(
        s3_client,
        bucket=bucket,
        key=file_path
    )
    
    
    print(f"✅ TAG_NAME: {tag_name}")
    df = df[df["tag_name"]==tag_name]
    print(df.shape)
    
    print(f"✅ KMeans Endpoint: {endpoint_kmeans}")
    print(f"✅ RCF Endpoint: {endpoint_rcf}")

if __name__ == "__main__":
    main()
