import argparse
from logic.training_helper import run_training

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bucket", type=str, required=True)
    parser.add_argument("--input_key", type=str, required=True)
    parser.add_argument("--model_output_key", type=str, required=True)
    parser.add_argument("--clustered_output_key", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()

    run_training(
        bucket=args.input_bucket,
        input_key=args.input_key,
        model_output_key=args.model_output_key,
        clustered_output_key=args.clustered_output_key,
        n_clusters=args.n_clusters
    )

if __name__ == "__main__":
    main()
