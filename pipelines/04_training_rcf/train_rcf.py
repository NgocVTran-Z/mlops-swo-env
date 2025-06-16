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

    local_input = "/opt/ml/processing/input/input_file.csv"
    local_output_model = "/opt/ml/processing/output/model.tar.gz"

    # Download file from S3
    rcf_helper.download_from_s3(input_bucket, input_key, local_input)

    # Load data
    df = pd.read_csv(local_input)
    print(f"✅ Loaded input data: {df.shape}")

    # Train RCF model
    model = rcf_helper.train_rcf(df)

    # Save model tar.gz
    joblib.dump(model, "model.joblib")
    os.system("tar -czvf model.tar.gz model.joblib")
    os.rename("model.tar.gz", local_output_model)
    print("✅ Model saved:", local_output_model)

if __name__ == "__main__":
    main()
