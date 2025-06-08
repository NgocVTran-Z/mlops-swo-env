import os
import json
import boto3
import pandas as pd
from io import BytesIO

def load_parquet_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))

def save_parquet_to_s3(df, s3_client, bucket, key):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

def main():
    print("✅ SageMaker preprocessing script started.")

    # Load environment variables
    bucket = os.environ["S3_BUCKET"]
    input_files = json.loads(os.environ["INPUT_FILES"])
    data_prefix = os.environ["DATA_PREFIX"]
    output_prefix = "mlops/pipelines/01_preprocessing_kmeans/"

    s3 = boto3.client("s3")

    for file_path in input_files:
        input_key = os.path.join(data_prefix, file_path)
        filename = os.path.basename(file_path)
        output_key = os.path.join(output_prefix, f"{filename.replace('.parquet', '')}_processed.parquet")

        print(f"📥 Loading: s3://{bucket}/{input_key}")
        df = load_parquet_from_s3(s3, bucket, input_key)

        if "value" not in df.columns:
            print(f"⚠️ Skipping {filename} — no 'value' column found.")
            continue

        # Apply transformation
        df["value"] = df["value"] * 10
        print(f"🔍 Preview of {filename}:\n", df.head())

        # Take top 5 rows
        df_head = df.head(15)

        # Save to S3
        print(f"📤 Saving to: s3://{bucket}/{output_key}")
        save_parquet_to_s3(df_head, s3, bucket, output_key)

    print("✅ SageMaker preprocessing completed.")
