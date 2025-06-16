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
        # Parse input body
        body = json.loads(event.get("body", "{}"))
        input_s3_uri = body.get("input_s3_uri")
        tag_name = body.get("tag_name")

        if not input_s3_uri or not tag_name:
            raise ValueError("Missing required fields: input_s3_uri or tag_name")

        # Load env vars
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]
        output_prefix = os.environ["OUTPUT_PREFIX"]
        endpoint_name = os.environ["ENDPOINT_NAME"]

        # Generate unique job name
        job_name = f"inference-rcf-{uuid.uuid4().hex[:8]}"

        # Submit SageMaker processing job
        sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=role_arn,
            AppSpecification={
                "ImageUri": image_uri,
                "ContainerEntrypoint": [
                    "python3",
                    "/opt/ml/processing/code/pipelines/05_inference_rcf/inference_rcf.py"
                ]
            },
            ProcessingInputs=[
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": f"s3://{bucket}/{code_prefix}",
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "output-1",
                        "S3Output": {
                            "S3Uri": f"s3://{bucket}/{output_prefix}",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            Environment={
                "S3_BUCKET": bucket,
                "INPUT_S3_URI": input_s3_uri,
                "TAG_NAME": tag_name,
                "ENDPOINT_NAME": endpoint_name
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeSizeInGB": 30
                }
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "SageMaker inference job started", "job_name": job_name})
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
