import os
from logic.training_helper import run_training
from sagemaker.sklearn.model import SKLearnModel
    

def load_env_inputs():
    """
    Load environment variables and parse input.
    Returns:
        input_files (List[str]): list of S3 file paths
        n_clusters (int): number of clusters
        tracking_uri (str): MLflow tracking URI
    """
    print("Loading environment variables...")

    input_files_str = os.environ.get("INPUT_FILES", "")
    n_clusters = int(os.environ.get("N_CLUSTERS", "5"))
    tracking_uri = os.environ.get("TRACKING_SERVER_ARN", "")

    input_files = input_files_str.split("|") if input_files_str else []

    print("INPUT_FILES raw string:", input_files_str)
    print("Parsed input_files:")
    for f in input_files:
        print(" -", f)

    print("N_CLUSTERS:", n_clusters)
    print("TRACKING_SERVER_ARN:", tracking_uri)

    return input_files, n_clusters, tracking_uri


def main():
    input_files, n_clusters, tracking_uri = load_env_inputs()
    print("input files type:", type(input_files))
    print("input files:", input_files)
    print("n cluster:", n_clusters)
    print("Environment loaded successfully.")

    # train kmeans model
    bucket = os.environ["S3_BUCKET"]
    model_output_key = "" # s3 bucket s3://swo-ngoctran-public/mlops/pipelines/02_training_kmeans/
    run_training(bucket=bucket, 
                 input_files=input_files,
                 n_clusters=n_clusters,
                 model_output_key=model_output_key,
                 tracking_uri=tracking_uri
                )

    # inference
    sk_model = SKLearnModel(
        model_data=f"s3://{bucket}/{s3_model_key}",
        role=role,
        entry_point="inference.py",  
        framework_version="0.23-1",  
        sagemaker_session=session
    )
    
    predictor = sk_model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )
    print(f"Endpoint deployed: {endpoint_name}")

    

if __name__ == "__main__":
    main()
