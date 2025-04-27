# from flask import Flask, request, jsonify, abort
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# with open("model_custom.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route("/")
# def home():
#     return "Housing ML Model is Running"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     # Input validation
#     if "features" not in data:
#         abort(400, description="'features' key is missing.")

#     features = data["features"]

#     # Handle multiple inputs
#     if isinstance(features[0], list):  # list of inputs
#         if not all(len(f) == 4 for f in features):
#             abort(400, description="Each input must have 4 float values.")
#         predictions = model.predict(features).tolist()
#         return jsonify({"predictions": predictions})
#     else:  # single input
#         if len(features) != 4:
#             abort(400, description="Input must have  float values.")
#         input_array = np.array(features).reshape(1, -1)
#         prediction = int(model.predict(input_array)[0])
#         confidence = float(model.predict_proba(input_array)[0][prediction])
#         return jsonify({"prediction": prediction, "confidence": confidence})

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=9000)

from flask import Flask, request, jsonify, abort
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained housing model
with open("model_custom.pkl", "rb") as f:
    housing_model = pickle.load(f)

@app.route("/")
def home():
    return "Housing Price Prediction Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Input validation
    if "features" not in data:
        abort(400, description="'features' key is missing.")

    features = data["features"]

    # Handle multiple inputs
    if isinstance(features[0], list):  # list of inputs
        if not all(len(f) == housing_model.n_features_in_ for f in features):
            abort(400, description=f"Each input must have {housing_model.n_features_in_} float values.")
        predictions = housing_model.predict(features).tolist()
        return jsonify({"predictions": predictions})
    else:  # single input
        if len(features) != housing_model.n_features_in_:
            abort(400, description=f"Input must have {housing_model.n_features_in_} float values.")
        input_array = np.array(features).reshape(1, -1)
        prediction = float(housing_model.predict(input_array)[0])
        return jsonify({"predicted_price": prediction})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "Housing Price Regression",
        "features": housing_model.n_features_in_,
        "model": str(housing_model.named_steps['regressor'])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)