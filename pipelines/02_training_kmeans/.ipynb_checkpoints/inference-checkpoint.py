# %%writefile inference.py
import joblib
import numpy as np

def model_fn(model_dir):
    model = joblib.load(f"{model_dir}/model.joblib")
    return model


# 1 line prediction 
# def input_fn(request_body, request_content_type):
#     if request_content_type == "text/csv":
#         data = np.array([list(map(float, request_body.strip().split(",")))])
#         return data
#     raise ValueError(f"Unsupported content type: {request_content_type}")

# >1 line
def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        lines = request_body.strip().split("\n")
        data = np.array([[float(x)] for x in lines])
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")



def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return str(prediction.tolist())
