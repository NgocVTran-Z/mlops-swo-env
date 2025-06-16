import os
import json
import uuid
import boto3
import urllib.parse

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("🟢 Lambda triggered.")
    print("📨 Incoming event:", event)

    try:
        body = json.loads(event.get("body", "{}"))

        analog_tag = body.get("analog_tag")
        motor = body.get("motor")
        input_s3_uri = body.get("input_s3_uri")

        if not analog_tag or not motor or not input_s3_uri:
            raise ValueError("Missing one of required input parameters.")

        # Parse input_s3_uri to bucket and key
        parsed = urllib.parse.urlparse(input_s3_uri)
        input_bucket = parsed.netloc
        input_key = parsed.path.lstrip("/")

        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]

        job_name = f"training-rcf-{uuid.uuid4().hex[:8]}"
        output_uri = f"s3://{bucket}/mlops/models/rcf/{analog_tag}_{motor}_{uuid.uuid4().hex[:8]}/"

        print("Output URI:", output_uri)

        sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=role_arn,
            AppSpecification={
                "ImageUri": image_uri,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/pipelines/04_training_rcf/train_rcf.py"]
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
                            "S3Uri": output_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            Environment={
                "ANALOG_TAG": analog_tag,
                "MOTOR": motor,
                "INPUT_BUCKET": input_bucket,
                "INPUT_KEY": input_key
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 2,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 50
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
