import boto3
import joblib
from sagemaker import RandomCutForest

def download_from_s3(bucket, key, local_path):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)

def train_rcf(df):
    train_data = df["value"].values.reshape(-1, 1)
    rcf = RandomCutForest(num_samples_per_tree=512, num_trees=50)
    rcf.fit(train_data)
    return rcf
