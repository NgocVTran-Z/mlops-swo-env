import os
import json
import boto3
import pandas as pd
from io import BytesIO
import sys

#-------------debug
import sys
import os

print("="*40)
print("[DEBUG] sys.executable:", sys.executable)
print("[DEBUG] Python version:", sys.version)
print("[DEBUG] sys.path:")
for p in sys.path:
    print("   ", p)
print("[DEBUG] Current working dir:", os.getcwd())
print("="*40)

print("CWD:", os.getcwd())
print("Files in current dir:", os.listdir('.'))
print("Parent dir:", os.listdir('..'))
print("Repo root?", os.listdir('../../'))
print("="*40)

# Test import sagemaker
try:
    import sagemaker
    print("[DEBUG] sagemaker imported successfully:", sagemaker.__version__)
except Exception as e:
    print("[ERROR] Failed to import sagemaker:", str(e))
#---------end of debug


# Add shared module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))

from logic.preprocessing_helper import internal_preprocessing
from utils.general_utils import load_parquet_from_s3  # Only keep load, not save

import mlflow
print("✅ mlflow imported:", mlflow.__version__)


def main():
    print("✅ SageMaker preprocessing kmeans script started ...")

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
            
    
            # Embed tag into the output filename
            output_filename = f"{filename.replace('.parquet', '')}_{tag}_processed.parquet"
            output_path = os.path.join(output_dir.rstrip("/"), input_subfolder, output_filename)
            
            # Ensure output folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            df_tag = internal_preprocessing(df, filename, tag, output_filename, output_path)

            output_paths = local_to_s3_path(local_path=output_path,
                                            bucket=bucket,
                                            output_prefix="mlops/pipelines/01_preprocessing_kmeans")
            print(f"Saving to: {output_paths}")
            df_tag.to_parquet(output_path, index=False)

    print("✅ SageMaker preprocessing kmeans completed.")


def local_to_s3_path(local_path, bucket, output_prefix):
    # Remove the base local prefix
    relative_path = local_path.replace("/opt/ml/processing/output/", "")
    return f"s3://{bucket}/{output_prefix.rstrip('/')}/{relative_path}"



if __name__ == "__main__":
    main()



    