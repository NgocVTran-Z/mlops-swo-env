#!/bin/bash

# Create folder structure
mkdir -p shared/utils
mkdir -p pipelines/01_preprocessing_kmeans/logic
mkdir -p pipelines/02_training_kmeans/logic
mkdir -p pipelines/03_preprocessing_rcf/logic
mkdir -p pipelines/04_training_rcf/logic
mkdir -p pipelines/05_inferencing_rcf/logic
mkdir -p shared/utils
mkdir -p config
mkdir -p lambda/trigger_preprocessing
mkdir -p .github/workflows


# Create empty Python files
touch shared/utils/general_utils.py
touch pipelines/01_preprocessing_kmeans/logic/preprocessing_helper.py
touch pipelines/01_preprocessing_kmeans/preprocess_kmeans.py
touch pipelines/02_training_kmeans/training_kmeans.py
touch lambda/trigger_preprocessing/lambda_function.py

touch .github/workflows/sync-to-s3.yml




