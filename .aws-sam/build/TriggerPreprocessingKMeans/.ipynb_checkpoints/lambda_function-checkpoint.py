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
        folders = body.get("folders", [])
        if not folders or not isinstance(folders, list):
            raise ValueError("Missing or invalid 'folders' in request payload - testing")

        # Load env vars
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]
        data_prefix = os.environ["DATA_PREFIX"]

        # Collect all parquet files from all folders
        all_files = []
        for folder in folders:
            prefix_to_search = f"{data_prefix}{folder}/"
            print(f"🔍 Searching: s3://{bucket}/{prefix_to_search}")
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_to_search)
            files_in_folder = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".parquet")]
            all_files.extend(files_in_folder)

        # Convert to relative path (strip off data_prefix)
        relative_files = [key.replace(data_prefix, "") for key in all_files]

        if not relative_files:
            raise ValueError(f"No .parquet files found in specified folders: {folders}")

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
