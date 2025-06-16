from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import sys
import gc

import os
sys.path.append(os.path.abspath(".."))
from common import *
from params import *
from visualize import log_kde_chart


def training_kmean(
    filtered_speed: pd.DataFrame,
    tag_name: str,
    motor: str,
    
    # params to log
    lst_training_paths, 
    destination_parquet_folder, 
    filename,
    
    # model params
    n_clusters: int = 5,
    random_state: int = 42,
    experiment_name: str = experiment_name_kmean
) -> pd.DataFrame:

    # sort the input first
    assert "time_utc" in filtered_speed.columns and "value" in filtered_speed.columns
    filtered_speed = filtered_speed.sort_values("time_utc").reset_index(drop=True)

    mlflow.set_experiment(experiment_name)

    now = datetime.now()
    now = now.strftime("%d%m%Y_%HH%MM%SS")
    
    with mlflow.start_run(run_name=f"{tag_name}_{n_clusters}-cluster"):
        # --- Log basic info
        mlflow.set_tags({
            "motor": motor,
            "phase": "clustering",
            "run time": now
        })
        mlflow.log_param("tag_name", tag_name)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("row_count", filtered_speed.shape[0])

        mlflow.log_param("list of source training data", lst_training_paths)
        
        
        # --- Log overall KDE chart
        log_kde_chart(
            df=filtered_speed,
            title="Histogram + KDE - value distribution",
            filename=f"charts/kde_overall_{tag_name}.png",
            color="blue"
        )


        # --- Train KMeans
        # X = filtered_speed[["value"]].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        filtered_speed["speed_cluster"] = kmeans.fit_predict(filtered_speed[["value"]].values.reshape(-1, 1))

        cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
        mlflow.log_param("cluster_centers", cluster_centers.tolist())

        # --- Log cluster metrics
        cluster_counts = filtered_speed['speed_cluster'].value_counts().sort_index()
        for i, count in cluster_counts.items():
            mlflow.log_metric(f"cluster_{i}_count", count)

        for i in sorted(filtered_speed['speed_cluster'].unique()):
            # df_cluster = filtered_speed[filtered_speed["speed_cluster"] == i]
            stats = filtered_speed[filtered_speed["speed_cluster"] == i]["value"].describe().to_dict()
            
            rename_map = {
                "25%": "percentile_25",
                "50%": "percentile_50",
                "75%": "percentile_75"
            }
            for stat_name, val in stats.items():
                safe_name = rename_map.get(stat_name, stat_name)
                mlflow.log_metric(f"cluster_{i}_{safe_name}", val)

            
            mlflow.log_metric(f"cluster_{i}_skew", filtered_speed[filtered_speed["speed_cluster"] == i]["value"].skew())
            mlflow.log_metric(f"cluster_{i}_kurtosis", filtered_speed[filtered_speed["speed_cluster"] == i]["value"].kurtosis())

            log_kde_chart(
                df=filtered_speed[filtered_speed["speed_cluster"] == i],
                title=f"KDE - Cluster {i} - {tag_name}",
                filename=f"charts/kde_cluster_{i}_{tag_name}.png",
                color="green"
            )

        # --- Save full clustered dataset
        current_time = get_current_timestamp_string()
        filename = f"{tag_name}_{current_time}.parquet"
        save_df_to_s3_parquet(
            df=filtered_speed,
            s3_path=destination_parquet_folder,
            filename=filename
        )
        
        mlflow.log_param("saved result destination", destination_parquet_folder + filename)
        
        print(f"✅ KMeans clustering + MLflow logging complete for {tag_name}")

        
        

    

    return filtered_speed