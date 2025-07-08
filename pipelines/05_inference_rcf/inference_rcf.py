import os

def main():
    input_s3_uri = os.environ.get("INPUT_S3_URI")
    tag_name = os.environ.get("TAG_NAME")
    endpoint_kmeans = os.environ.get("ENDPOINT_NAME_KMEANS")
    endpoint_rcf = os.environ.get("ENDPOINT_NAME_RCF")

    print("✅ Inference RCF script started")
    print(f"📂 INPUT_S3_URI: {input_s3_uri}")
    print(f"🏷️ TAG_NAME: {tag_name}")
    print(f"🧠 KMeans Endpoint: {endpoint_kmeans}")
    print(f"🌲 RCF Endpoint: {endpoint_rcf}")

if __name__ == "__main__":
    main()
