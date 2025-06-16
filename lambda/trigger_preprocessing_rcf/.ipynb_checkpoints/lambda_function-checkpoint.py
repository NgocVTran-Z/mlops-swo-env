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

        analog_tag = body.get("analog_tag")
        motor = body.get("motor")
        input_bucket = body.get("input_bucket")
        input_key = body.get("input_key")

        if not analog_tag or not motor or not input_bucket or not input_key:
            raise ValueError("Missing one of required input parameters.")

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
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/pipelines/03_training_rcf/preprocessing_rcf.py"]
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
