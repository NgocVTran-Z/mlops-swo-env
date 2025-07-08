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

import json
import boto3
import pandas as pd
import numpy as np



def apply_shingling(series, window_size):
    """Convert 1D series into list of shingled vectors (sliding window)"""
    return [series[i:i+window_size].tolist() for i in range(len(series) - window_size + 1)]

def get_anomaly_scores(endpoint_name, shingled_vectors):
    """Call RCF endpoint with list of shingled feature vectors"""
    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "instances": [{"features": vec} for vec in shingled_vectors]
    }

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )

    result = response["Body"].read().decode("utf-8")
    scores = json.loads(result)
    return [item["score"] for item in scores]

def process_anomaly_scores(df, endpoint_name_rcf, window_size=5):
    # Step 1: Ensure "value" column is clean
    values = df["value"].dropna().astype(float).tolist()

    # Step 2: Shingling
    shingled = apply_shingling(values, window_size)

    if len(shingled) == 0:
        raise ValueError("Shingling resulted in no data. Check if 'value' has enough rows.")

    # Step 3: Call endpoint
    scores = get_anomaly_scores(endpoint_name_rcf, shingled)

    # Step 4: Padding to align score with original df
    padded_scores = [np.nan] * (window_size - 1) + scores
    df = df.reset_index(drop=True)
    df["anomaly_scores"] = padded_scores

    return df

def save_parquet_to_s3(df, bucket, output_prefix, filename="inference_output.parquet"):
    """Save DataFrame as .parquet to S3"""
    import io

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3 = boto3.client("s3")
    output_key = f"{output_prefix}{filename}"
    s3.upload_fileobj(buffer, bucket, output_key)

    print(f"✅ Saved output to s3://{bucket}/{output_key}")



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
    print("cluster_nr:", cluster_nr)
    # df = df[df["cluster_nr"]==cluster_nr]
    print("Filtered cluster number:", df.shape)
    
    
    print(f"✅ RCF Endpoint: {endpoint_rcf}")
    # Transform df - preprocessing data before put in RCF model and call MME 
    df = process_anomaly_scores(df, endpoint_rcf)
    print(df.shape)

    # save to inference bucket
    bucket = os.environ["S3_BUCKET"]
    output_prefix = os.environ["OUTPUT_PREFIX"]  # ends with `/`

    # Tách phần key tương đối sau "mlops/raw_data/"
    input_s3_uri = os.environ["INPUT_S3_URI"]
    raw_data_prefix = os.environ.get("DATA_PREFIX", "mlops/raw_data/")  # fallback 
    
    bucket, input_key = parse_s3_uri(input_s3_uri)
    relative_path = input_key.replace(raw_data_prefix, "")  # 2024-01/30/23/sample.parquet
    subfolder = "/".join(relative_path.split("/")[:-1])     # 2024-01/30/23
    
    # Full output_prefix
    output_prefix = os.environ["OUTPUT_PREFIX"]  # eg: mlops/pipelines/05_inference_rcf/
    final_output_prefix = f"{output_prefix}{subfolder}/"

    # save to s3 bucket
    save_parquet_to_s3(df, bucket, final_output_prefix)

    
    

if __name__ == "__main__":
    main()
