import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor
from sagemaker import get_execution_role

import mlflow


def get_pipeline():
    role = os.environ.get("SAGEMAKER_ROLE_ARN", get_execution_role())
    image_uri = os.environ["PROCESSING_IMAGE_URI"]

    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large"
    )

    step_process = ProcessingStep(
        name="TestProcessingStep",
        processor=processor,
        code="test_entrypoint.py",  # chỉ gọi 1 file duy nhất
        outputs=[]
    )

    return Pipeline(
        name="test-00-pipeline",
        steps=[step_process]
    )
