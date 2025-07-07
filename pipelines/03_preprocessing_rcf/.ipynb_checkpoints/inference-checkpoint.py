
import joblib
import os
import json
import pandas as pd



def model_fn(model_dir):
    model_path = os.path.join(model_dir, "kmeans_model.pkl")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    raise ValueError("Unsupported content type: {}".format(request_content_type))



def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds.tolist()

def output_fn(prediction, content_type):
    return json.dumps({"speed_cluster": prediction})
