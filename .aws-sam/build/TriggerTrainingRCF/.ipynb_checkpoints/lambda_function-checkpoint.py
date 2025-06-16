import os
import json
import uuid
import boto3
import datetime

sagemaker = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("🟢 Lambda triggered.")
    print("📨 Incoming event:", event)

    try:
        body = json.loads(event.get("body", "{}"))

        analog_tag = body.get("analog_tag")
        motor = body.get("motor")
        input_bucket = body.get("input_bucket")
        input_key = body.get("input_key")
        model_output_key = body.get("model_output_key")

        if not analog_tag or not motor or not input_bucket or not input_key or not model_output_key:
            raise ValueError("Missing one of required input parameters.")

        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        bucket = os.environ["S3_BUCKET"]
        code_prefix = os.environ["CODE_PREFIX"]

        job_name = f"training-rcf-{uuid.uuid4().hex[:8]}"

        print("Training job name:", job_name)

        sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn=role_arn,
            AlgorithmSpecification={
                "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/randomcutforest:1",
                "TrainingInputMode": "File"
            },
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{input_bucket}/{input_key}",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv"
                }
            ],
            OutputDataConfig={
                "S3OutputPath": f"s3://{input_bucket}/{model_output_key}"
            },
            ResourceConfig={
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 2,
                "VolumeSizeInGB": 50
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": 3600
            },
            EnableNetworkIsolation=True
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "SageMaker training job started", "job_name": job_name})
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
