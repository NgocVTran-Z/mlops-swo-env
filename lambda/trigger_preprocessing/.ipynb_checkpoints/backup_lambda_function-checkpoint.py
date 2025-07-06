import boto3
import os
import time
import traceback

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    try:
        print(" Lambda triggered.")
        print(" Event received:", event)

        # 1. get info s3
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]
        print(f"📂 File uploaded: s3://{bucket}/{key}")

        # 2. get env
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        script_s3 = os.environ["SCRIPT_S3_URI"]
        output_s3 = os.environ["OUTPUT_S3_PATH"]
        endpoint = os.environ["ENDPOINT_NAME"]

        # 3. Create job name
        timestamp = int(time.time())
        job_name = f"inference-job-{timestamp}"

        print(f" Starting ProcessingJob: {job_name}")
        print(f" Input S3: s3://{bucket}/{key}")
        print(f" Script S3: {script_s3}")
        print(f" Output S3: {output_s3}")
        print(f" Endpoint: {endpoint}")
        print(f" Image URI: {image_uri}")

        # 4. Create SageMaker Processing Job
        response = sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=role_arn,
            AppSpecification={
                "ImageUri": image_uri,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/inference.py"]
            },
            ProcessingInputs=[
                {
                    "InputName": "input-data",
                    "S3Input": {
                        "S3Uri": f"s3://{bucket}/{key}",
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                },
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": script_s3,
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "output-data",
                        "S3Output": {
                            "S3Uri": output_s3,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeSizeInGB": 10
                }
            },
            Environment={
                "ENDPOINT_NAME": endpoint
            }
        )

        print(f" Job submitted: {job_name}")
        print(" SageMaker response:", response)

        return {
            "status": "processing_started",
            "job_name": job_name,
            "input_file": key
        }

    except Exception as e:
        print(" Exception occurred:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }
