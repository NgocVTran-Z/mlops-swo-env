import boto3
import json
import os

sagemaker_client = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("Event received:", event)
    
    # Extract input parameters from the API payload
    input_bucket = event["input_bucket"]
    input_key = event["input_key"]
    model_output_key = event["model_output_key"]
    clustered_output_key = event["clustered_output_key"]
    n_clusters = event.get("n_clusters", 5)

    processing_job_name = f"training-kmeans-job-{context.aws_request_id}"

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
                "/opt/ml/processing/code/pipelines/02_training_kmeans/train_kmeans.py"
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
            "--input_bucket", input_bucket,
            "--input_key", input_key,
            "--model_output_key", model_output_key,
            "--clustered_output_key", clustered_output_key,
            "--n_clusters", str(n_clusters)
        ]
    )

    print("Started SageMaker Processing Job:", processing_job_name)

    return {
        "statusCode": 200,
        "body": json.dumps({"processing_job_name": processing_job_name})
    }
