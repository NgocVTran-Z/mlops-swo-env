import argparse
import os
import pandas as pd
import boto3
import json
from pipelines.05_inference_rcf.logic import inference_helper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_bucket', type=str, default=os.environ.get('S3_BUCKET'))
    parser.add_argument('--input_s3_uri', type=str, default=os.environ.get('INPUT_S3_URI'))
    parser.add_argument('--tag_name', type=str, default=os.environ.get('TAG_NAME'))
    parser.add_argument('--endpoint_name', type=str, default=os.environ.get('ENDPOINT_NAME'))
    return parser.parse_args()

def main():
    args = parse_args()

    # Download input CSV from S3
    input_path = '/opt/ml/processing/input/input_file.csv'
    os.makedirs(os.path.dirname(input_path), exist_ok=True)

    s3 = boto3.client("s3")
    bucket, key = inference_helper.parse_s3_uri(args.input_s3_uri)
    s3.download_file(bucket, key, input_path)

    # Load data
    df = pd.read_csv(input_path)

    # Call endpoint to get anomaly scores
    scores = inference_helper.predict_batch_rcf(df, args.endpoint_name)

    # Append results
    df['anomaly_score'] = scores
    df['tag_name'] = args.tag_name

    # Write output
    output_dir = '/opt/ml/processing/output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Inference result written to {output_file}")

if __name__ == "__main__":
    main()
