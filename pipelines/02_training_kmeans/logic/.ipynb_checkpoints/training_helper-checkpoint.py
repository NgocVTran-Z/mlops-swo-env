import pandas as pd
import boto3
import joblib
from sklearn.cluster import KMeans
from shared.utils.general_utils import read_parquet_from_s3

# Add shared module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))

from utils.params import mapping_tags



def save_model_to_s3(model, bucket, model_key):
    joblib.dump(model, "/tmp/model.joblib")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/model.joblib", bucket, model_key)

def save_dataframe_to_s3(df, bucket, output_key):
    out_buffer = df.to_parquet(index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=output_key, Body=out_buffer)


def training_kmean(
    filtered_speed: pd.DataFrame,
    tag_name: str,
    motor: str,
    
    # params to log
    lst_training_paths, 
    destination_parquet_folder=False, 
    filename=False,
    
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

def run_training(bucket, input_files, n_clusters, model_output_key, tracking_uri):
    print("Loading data from S3...")
    df = read_parquet_from_s3(bucket, input_files)

    # Get mapping tag names
    digital_tag = mapping_tags["Digital"][tag]
    speed_tag = mapping_tags["Speed"][tag]
    
    training_kmean(
        filtered_speed=df,
        tag_name=digital_tag,
        motor=speed_tag,
        
        # params to log
        lst_training_paths=model_output_key, 
        # destination_parquet_folder, 
        # filename,
        
        # model params
        n_clusters=n_clusters,
        # random_state=42,
        experiment_name="pipeline_02_training_kmeans"
    ) -> pd.DataFrame:

    print("Training pipeline completed.")

