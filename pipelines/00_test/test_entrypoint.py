import mlflow
import sagemaker

import joblib
from sagemaker import RandomCutForest
from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path

# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np


from logic.test_import import test_import_function


def run_test_logic():
    print("🔧 Running test logic...")
    print("🔧 Test import sagemaker...")
    print("something wrong ...")
    # tracking_uri = "https://t-shqyllgofmce.us-east-1.experiments.sagemaker.aws"
    # mlflow.set_tracking_uri(tracking_uri)
    # mlflow.set_experiment("00_test_pipeline")

    # with mlflow.start_run():
    #     mlflow.log_param("test_param", 123)
    #     mlflow.log_metric("test_metric", 0.98)

    # print("✅ MLflow logging complete.")

if __name__ == "__main__":
    run_test_logic()
