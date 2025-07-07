import os

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
    print("input files:", input_files)
    print("n cluster:", n_clusters)
    print("Environment loaded successfully.")


if __name__ == "__main__":
    main()
