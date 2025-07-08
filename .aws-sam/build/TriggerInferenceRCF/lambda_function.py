import os
import json
import uuid
import boto3

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("🟢 Lambda triggered.")
    print("📨 Incoming event:", event)

    try:
        # Parse input body
        body = json.loads(event.get("body", "{}"))
        input_s3_uri = body.get("input_s3_uri")
        tag_name = body.get("tag_name")
        endpoint_name_kmeans = body.get("endpoint_name_kmeans")
        endpoint_name_rcf = body.get("endpoint_name_rcf")

        if not input_s3_uri or not tag_name or not endpoint_name_kmeans or not endpoint_name_rcf:
            raise ValueError("Missing required fields in input JSON.")

        # Load env vars
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]
        output_prefix = os.environ["OUTPUT_PREFIX"]

        job_name = f"inference-rcf-{uuid.uuid4().hex[:8]}"

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
                            "S3Uri": f"s3://{bucket}/{output_prefix}{job_name}/",
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
                "ENDPOINT_NAME_KMEANS": endpoint_name_kmeans,
                "ENDPOINT_NAME_RCF": endpoint_name_rcf
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
