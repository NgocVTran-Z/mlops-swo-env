import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow



def log_kde_chart(df: pd.DataFrame, title: str, filename: str, color: str = "blue"):
    """
    Vẽ biểu đồ KDE và log vào MLflow dưới dạng figure.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df["value"], bins=30, kde=True, color=color, alpha=0.5)
    plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), filename)
    plt.close()