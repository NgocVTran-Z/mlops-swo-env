import os
import pandas as pd
import joblib
from shared.logic import rcf_helper

def main():
    print("🟢 Starting RCF training pipeline.")

    analog_tag = os.environ.get("ANALOG_TAG")
    motor = os.environ.get("MOTOR")
    input_bucket = os.environ.get("INPUT_BUCKET")
    input_key = os.environ.get("INPUT_KEY")

    local_input = "/opt/ml/processing/input/input_file.parquet"
    local_output_model = "/opt/ml/processing/output/model.joblib"

    # Download parquet from S3
    rcf_helper.download_from_s3(input_bucket, input_key, local_input)

    # Load data
    df = pd.read_parquet(local_input)
    print(f"✅ Loaded input data: {df.shape}")

    # Train RCF
    model = rcf_helper.train_rcf(df)

    # Save model
    joblib.dump(model, local_output_model)
    print("✅ Model saved:", local_output_model)

if __name__ == "__main__":
    main()
