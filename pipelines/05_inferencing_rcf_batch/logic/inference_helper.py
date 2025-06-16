import boto3
import numpy as np
import io
import pandas as pd
from urllib.parse import urlparse

def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def predict_batch_rcf(df, endpoint_name):
    runtime = boto3.client('sagemaker-runtime')

    # Convert dataframe to CSV string format
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)
    payload = csv_buffer.getvalue()

    # Call endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )

    result = json.loads(response['Body'].read().decode('utf-8'))
    scores = result['scores']

    return scores
