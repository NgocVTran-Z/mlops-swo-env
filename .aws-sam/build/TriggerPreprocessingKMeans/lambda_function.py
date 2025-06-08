import boto3
import json
import uuid
import os

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    body = json.loads(event["body"]) if "body" in event else event
    input_files = body.get("input_files", ["2024-01/sample.parquet"])
    job_name = f"preprocess-kmeans-{uuid.uuid4().hex[:8]}"

    sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        AppSpecification={
            "ImageUri": os.environ["PROCESSING_IMAGE_URI"],
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/preprocess_kmeans.py"]
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{os.environ['S3_BUCKET']}/{os.environ['CODE_PREFIX']}",
                    "LocalPath": "/opt/ml/processing/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        ProcessingOutputConfig={"Outputs": []},
        Environment={
            "INPUT_FILES": json.dumps(input_files)
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 10
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": 600}
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": f"Started processing job {job_name}"})
    }
