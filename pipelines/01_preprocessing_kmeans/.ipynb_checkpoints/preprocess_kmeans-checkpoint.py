import os
import json
import boto3
import pandas as pd
from io import BytesIO
import sys

# Add shared module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))

from logic.preprocessing_helper import internal_preprocessing
from utils.general_utils import load_parquet_from_s3  # Only keep load, not save

def main():
    print("✅ SageMaker preprocessing script started ...")

    # Load environment variables
    bucket = os.environ["S3_BUCKET"]
    input_files = json.loads(os.environ["INPUT_FILES"])
    data_prefix = os.environ["DATA_PREFIX"]

    speed_tag = json.loads(os.environ.get("SPEED_TAG", "[]"))
    print(f"Speed tags selected: {speed_tag} !!!")
    for tag in speed_tag:
        print(tag)
    
    # Output directory for SageMaker to auto-upload to S3
    output_dir = "/opt/ml/processing/output"

    s3 = boto3.client("s3")

    for file_path in input_files:
        input_key = os.path.join(data_prefix, file_path)
        filename = os.path.basename(file_path)
        input_subfolder = os.path.dirname(file_path)
    
        s3 = boto3.client("s3")
        print(f"📥 Loading: s3://{bucket}/{input_key}")
        df = load_parquet_from_s3(s3, bucket, input_key)
    
        if "value" not in df.columns:
            print(f"Skipping {filename} — no 'value' column found.")
            continue
    
        for tag in speed_tag:
            df_tag = internal_preprocessing(df, filename, tag)
    
            # Embed tag into the output filename
            output_filename = f"{filename.replace('.parquet', '')}_{tag}_processed.parquet"
            output_path = os.path.join(output_dir.rstrip("/"), input_subfolder, output_filename)
    
            # Ensure output folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
            print(f"Saving to: {output_path}")
            df_tag.to_parquet(output_path, index=False)

    print("✅ SageMaker preprocessing completed.")



if __name__ == "__main__":
    main()



    