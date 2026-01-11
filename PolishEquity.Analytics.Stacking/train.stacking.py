import os
import pandas as pd
import requests
import joblib
from sklearn.linear_model import LogisticRegression

# Remote Infrastructure Configuration
STACKING_INPUT_PATH = "Data/stacking_input.csv"
EXTERNAL_MODEL_PATH = "https://external_model:8000/run_pipeline"
MODEL_SAVE_PATH = "Models/meta_learner.joblib"

def main():
    """
    Orchestrates the training of a meta-learner by integrating predictions 
    from the .NET LightGBM base model and the Python Random Forest API.
    """
    # 1. Validation of input data availability
    if not os.path.exists(STACKING_INPUT_PATH):
        print(f"[ERROR] Stacking source file not found at: {STACKING_INPUT_PATH}")
        return

    print(f"[INFO] Loading stacking data from {STACKING_INPUT_PATH}...")
    data_input = pd.read_csv(STACKING_INPUT_PATH)
    external_scores = []

    # 2. Remote Inference Loop (Gathering Python Model Predictions)
    print(f"[INFO] Fetching predictions from FastAPI endpoint...")
    for index, row in data_input.iterrows():
        try:
            # Map row to dictionary to serve as JSON payload
            payload = row.to_dict()
            response = requests.post(EXTERNAL_MODEL_PATH, json=payload, timeout=10)
            response.raise_for_status()

            prediction = response.json()

            # Handle different response formats (Direct list vs. Dict wrapped)
            # Standardizes the probability vector for the Meta-Learner
            val = prediction if isinstance(prediction, list) else prediction.get("probability", prediction)
            external_scores.append(val)

        except Exception as e:
            print(f"[WARNING] Inference failed for row {index}: {e}")
            # Inject a neutral distribution fallback to maintain dataset alignment
            external_scores.append([0.0] * 3) 

    # 3. Meta-Feature Engineering (Assembling Meta_X)
    print("[INFO] Aligning features from .NET and Python models...")
    
    # ML.NET flattens the 'Score' vector into 'Score.0', 'Score.1', etc. during SaveAsText
    df_net_features = data_input.filter(like="Score")

    # Wrap external model outputs into a structured DataFrame
    df_ext_features = pd.DataFrame(external_scores).add_prefix("Ext_Score_")

    # Concatenate base model outputs into the final Feature Matrix for the Meta-Learner
    Meta_X = pd.concat([
        df_net_features.reset_index(drop=True), 
        df_ext_features.reset_index(drop=True)
    ], axis=1)

    # 4. Target Label Extraction
    Meta_target = data_input["Label"]

    # 5. Meta-Learner Training (The 'Judge' Model)
    print(f"[INFO] Training Meta-Learner (Logistic Regression) on {Meta_X.shape[1]} input features.")
    
    # Logistic Regression is used here to find the optimal weights for each base model's opinion
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(Meta_X, Meta_target)

    # 6. Model Persistence
    joblib.dump(meta_learner, MODEL_SAVE_PATH)
    
    print(f"[SUCCESS] Meta-learner training complete. Saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()