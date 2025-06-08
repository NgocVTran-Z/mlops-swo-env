import os
import json
import boto3
import pandas as pd
from io import BytesIO



import sys

# path that import can be shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))

from utils.general_utils import load_parquet_from_s3, save_parquet_to_s3
from logic.preprocessing_helper import internal_preprocessing


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
        df = internal_preprocessing(df, filename)

        # Take top 5 rows
        df_head = df.head(3)

        # Save to S3
        print(f"📤 Saving to: s3://{bucket}/{output_key}")
        save_parquet_to_s3(df_head, s3, bucket, output_key)

    print("✅ SageMaker preprocessing completed.")
