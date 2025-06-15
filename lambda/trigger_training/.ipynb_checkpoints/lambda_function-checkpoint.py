import os
import json
import uuid
import boto3

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("🟢 Lambda triggered.")
    print("📨 Incoming event:", event)

    try:
        body = json.loads(event.get("body", "{}"))

        # Parse input from request payload
        input_bucket = body.get("input_bucket")
        input_key = body.get("input_key")
        model_output_key = body.get("model_output_key")
        clustered_output_key = body.get("clustered_output_key")
        n_clusters = body.get("n_clusters", 5)

        if not input_bucket or not input_key or not model_output_key or not clustered_output_key:
            raise ValueError("Missing one of required input parameters.")

        # Load env vars
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        image_uri = os.environ["PROCESSING_IMAGE_URI"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]

        # Submit SageMaker processing job
        job_name = f"training-kmeans-{uuid.uuid4().hex[:8]}"

        sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=role_arn,
            AppSpecification={
                "ImageUri": image_uri,
                "ContainerEntrypoint": [
                    "python3",
                    "/opt/ml/processing/code/pipelines/02_training_kmeans/train_kmeans.py"
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
                            "S3Uri": f"s3://{bucket}/{code_prefix}/dummy-output/",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            Environment={
                "INPUT_BUCKET": input_bucket,
                "INPUT_KEY": input_key,
                "MODEL_OUTPUT_KEY": model_output_key,
                "CLUSTERED_OUTPUT_KEY": clustered_output_key,
                "N_CLUSTERS": str(n_clusters)
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
