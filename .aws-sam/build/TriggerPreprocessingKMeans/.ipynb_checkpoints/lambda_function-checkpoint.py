import boto3
import json
import os

sagemaker_client = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("Event received:", event)
    
    if 'body' in event:
        try:
            event = json.loads(event['body'])
        except Exception as e:
            print("Failed to parse body:", e)

    folders = event["folders"]
    speed_tag = event.get("speed_tag", [])

    processing_job_name = f"preprocessing-kmeans-job-{context.aws_request_id}"

    response = sagemaker_client.create_processing_job(
        ProcessingJobName=processing_job_name,
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 30
            }
        },
        AppSpecification={
            "ImageUri": os.environ["PROCESSING_IMAGE_URI"],
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/code/pipelines/01_preprocessing_kmeans/preprocess_kmeans.py"
            ]
        },
        Environment={
            "PYTHONUNBUFFERED": "TRUE"
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": os.environ["CODE_S3_URI"],
                    "LocalPath": "/opt/ml/processing/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        ProcessingOutputConfig={
            "Outputs": []
        },
        Arguments=[
            "--folders", ",".join(folders),
            "--speed_tag", ",".join(speed_tag)
        ]
    )

    print("Started SageMaker Processing Job:", processing_job_name)

    return {
        "statusCode": 200,
        "body": json.dumps({"processing_job_name": processing_job_name})
    }
