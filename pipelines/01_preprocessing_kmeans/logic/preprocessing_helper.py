import pandas as pd


def internal_preprocessing(df, filename):
    print("🧠 This is internal logic of preprocessing pipeline")
    df["value"] = df["value"] * 200
    print(f"🔍 Preview of {filename}:\n", df.head())
    return df
