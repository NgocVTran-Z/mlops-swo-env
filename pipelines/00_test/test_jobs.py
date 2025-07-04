from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
import os

import mlflow
import sagemaker


def get_extra_step():
    print("test jobs")

    role = "arn:aws:iam::975049948583:role/LambdaSageMakerInferencePolicy"
    image_uri = "975049948583.dkr.ecr.us-east-1.amazonaws.com/sagemaker-custom-mlflow:latest"
    
    processor = ScriptProcessor(
        image_uri=image_uri,
        # os.environ["PROCESSING_IMAGE_URI"],
        command=["python3"],
        role=role,
        # os.environ["SAGEMAKER_ROLE_ARN"],
        instance_count=1,
        instance_type="ml.m5.large"
    )

    
    step = ProcessingStep(
        name="ExtraProcessingStep",
        processor=processor,
        code="test_entrypoint.py"
    )

    return step
    # return



