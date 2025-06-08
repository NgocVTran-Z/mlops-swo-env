import os
import json
import uuid
import boto3

sagemaker = boto3.client("sagemaker")
s3 = boto3.client("s3")

def lambda_handler(event, context):
    print("🟢 Lambda triggered.")
    print("📨 Incoming event:", event)

    try:
        body = json.loads(event.get("body", "{}"))
        folder = body.get("folder")  # e.g., "2024-01"
        if not folder:
            raise ValueError("Missing 'folder' in request payload")

        # Load env vars
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]
        data_prefix = os.environ["DATA_PREFIX"]

        # List all .parquet files in that folder
        prefix_to_search = f"{data_prefix}{folder}/"
        print(f"🔍 Listing .parquet files under: s3://{bucket}/{prefix_to_search}")

        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_to_search)
        all_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".parquet")]

        # Convert back to relative paths
        relative_files = [key.replace(data_prefix, "") for key in all_files]

        if not relative_files:
            raise ValueError(f"No .parquet files found in s3://{bucket}/{prefix_to_search}")

        print(f"✅ Found files: {relative_files}")

        # Submit SageMaker processing job
        job_name = f"preprocess-kmeans-{uuid.uuid4().hex[:8]}"
        script_uri = f"s3://{bucket}{code_prefix}preprocessing_kmeans.py"
        output_uri = f"s3://{bucket}/inference_result/"

        response = sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=role_arn,
            AppSpecification={
                "ImageUri": image_uri,
                "ScriptUri": script_uri
            },
            ProcessingInputs=[],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "output-1",
                        "S3Output": {
                            "S3Uri": output_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            Environment={
                "S3_BUCKET": bucket,
                "INPUT_FILES": json.dumps(relative_files),
                "DATA_PREFIX": data_prefix
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeSizeInGB": 20
                }
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "SageMaker processing job started", "job_name": job_name})
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
