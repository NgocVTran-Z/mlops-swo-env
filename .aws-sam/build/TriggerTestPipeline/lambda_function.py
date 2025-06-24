import os
import boto3

def lambda_handler(event, context):
    pipeline_name = "test-00-pipeline"  # Phải khớp với Pipeline trong test_pipeline.py

    sagemaker_client = boto3.client("sagemaker")

    try:
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name
        )

        return {
            "statusCode": 200,
            "body": f"Pipeline {pipeline_name} triggered. Execution ARN: {response['PipelineExecutionArn']}"
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": str(e)
        }
