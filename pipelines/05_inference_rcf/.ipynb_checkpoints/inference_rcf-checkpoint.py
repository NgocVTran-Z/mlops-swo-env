import os
import boto3
import pandas as pd
from io import BytesIO


import sys
print("PYTHONPATH:", sys.path)
print("Files in working dir:", os.listdir("/opt/ml/processing/code"))

# KMeans
# import sagemaker
# print(sagemaker.__version__)
# from sagemaker.predictor import Predictor
# from sagemaker.serializers import CSVSerializer




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
    cluster_nr = os.environ.get("CLUSTER_NR")

    print("✅ Inference RCF script started")
    print(f"✅ INPUT_S3_URI: {file_path}")

    # Parse S3 URI to get bucket and key
    bucket, key = parse_s3_uri(file_path)

    # Init boto3 S3 client
    s3_client = boto3.client("s3")

    # Load the parquet file from S3
    df = load_parquet_from_s3(s3_client, bucket, key)

    # Filter by tag_name
    print(f"✅ TAG_NAME: {tag_name}")
    df = df[df["tag_name"] == tag_name].copy()
    print(f" Data shape after filtering: {df.shape}")

    # Placeholder for next step
    print(f"✅ KMeans Endpoint: {endpoint_kmeans}")
    # # Extract 'value' column and dropna
    # values = df["value"].dropna().astype(float).tolist()
    # if len(values) == 0:
    #     raise ValueError("No values found in column 'value' after filtering.")
    # # Prepare CSV payload
    # csv_payload = "\n".join([str(x) for x in values])
    # # Init predictor
    # predictor = Predictor(
    #     endpoint_name=endpoint_kmeans,
    #     serializer=CSVSerializer(),
    #     deserializer=JSONDeserializer()
    # )
    # # Predict clusters
    # response = predictor.predict(csv_payload)  # Expect list of {"predicted_label": int}
    # cluster_preds = [item["predicted_label"] for item in response]
    # # Add back to dataframe
    # df = df.reset_index(drop=True)  # Ensure alignment
    # df["cluster_nr"] = cluster_preds
    
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
    cluster_results = []
    for val in df["value"]:
        payload = str(val)
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_kmeans,
            ContentType="text/csv",
            Body=payload
        )
        # Convert kết quả từ chuỗi về số
        pred_str = response["Body"].read().decode("utf-8")
        pred = int(eval(pred_str)[0])  # ví dụ "[2]" → 2
        cluster_results.append(pred)

    
    df["cluster_nr"] = cluster_results
    print("✅ Assigned clusters added to dataframe.")
    # print(df.head())
    df = df[df["cluster_nr"]==cluster_nr]
    print("Filtered cluster number:", df.shape)
    
    
    
    print(f"✅ RCF Endpoint: {endpoint_rcf}")
    # Transform df - preprocessing data before put in RCF model

    # Call MME
    
    
    

if __name__ == "__main__":
    main()
