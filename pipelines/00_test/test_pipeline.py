# import os
# from sagemaker.workflow.pipeline import Pipeline
# from sagemaker.workflow.steps import ProcessingStep
# from sagemaker.processing import ScriptProcessor
# from sagemaker import get_execution_role

# import mlflow
# import sagemaker
# import boto3
# from sagemaker.estimator import Estimator
# from sagemaker.inputs import TrainingInput

# import matplotlib.pyplot as plt
# import seaborn as sns

# from sagemaker.model import Model
# from sagemaker.transformer import Transformer

# from test_jobs import get_extra_step




# def get_pipeline():
#     role = os.environ.get("SAGEMAKER_ROLE_ARN", get_execution_role())
#     image_uri = os.environ["PROCESSING_IMAGE_URI"]

#     processor = ScriptProcessor(
#         image_uri=image_uri,
#         command=["python3"],
#         role=role,
#         instance_count=1,
#         instance_type="ml.m5.large"
#     )

#     # step_process = ProcessingStep(
#     #     name="TestProcessingStep",
#     #     processor=processor,
#     #     code="test_entrypoint.py",  # chỉ gọi 1 file duy nhất
#     #     outputs=[]
#     # )

#     return Pipeline(
#         name="test-00-pipeline",
#         steps=[
#             # step_process, 
#             get_extra_step()
#         ]
#     )


from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor

def get_pipeline():
    role = "arn:aws:iam::975049948583:role/LambdaSageMakerInferencePolicy"
    image_uri = "975049948583.dkr.ecr.us-east-1.amazonaws.com/sagemaker-custom-mlflow:latest"

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
        code="test_entrypoint.py",
        outputs=[]
    )

    return Pipeline(
        name="test-00-pipeline",
        steps=[step_process]
    )
