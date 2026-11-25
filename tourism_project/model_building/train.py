"""
Model Building with Experimentation Tracking
--------------------------------------------
  * loads the train/test splits produced by prep.py (in CI they arrive as a
    workflow artifact),
  * builds a preprocessing + XGBoost pipeline,
  * tunes it with GridSearchCV,
  * logs EVERY parameter combination that was tried (as nested MLflow runs)
    plus the best parameters and test metrics (parent run),
  * evaluates the best model on train and test,
  * saves the best model to tourism_project/deployment/ so the pipeline can
    commit it to the repository for the Streamlit app.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- config
SPLIT_DIR = os.getenv("SPLIT_DIR", ".")
MODEL_DIR = os.getenv("MODEL_DIR", "tourism_project/deployment")
MODEL_NAME = "best_tourism_model_v1.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "tourism-wellness-package")
THRESHOLD = 0.5          # probability cut-off used for the final class label
RANDOM_STATE = 42

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# ---------------------------------------------------------------- data
Xtrain = pd.read_csv(os.path.join(SPLIT_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(SPLIT_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(SPLIT_DIR, "ytrain.csv")).squeeze("columns")
ytest = pd.read_csv(os.path.join(SPLIT_DIR, "ytest.csv")).squeeze("columns")
print(f"Train {Xtrain.shape} | Test {Xtest.shape}")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
    # ordinal-encoded in prep.py (0 = lowest tier / seniority)
    "ProductPitched", "Designation",
]
categorical_features = [   # nominal -> one-hot encoded
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
]

# ---------------------------------------------------------------- model
# Class imbalance (~19% buyers): weight the positive class by neg/pos ratio
scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [4, 6, 8],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__reg_lambda": [1.0],
}


def metrics(y_true, y_prob, prefix):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
    }


with mlflow.start_run(run_name="xgb_gridsearch") as parent:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # --- log every tuned parameter combination as a nested run
    res = grid.cv_results_
    for i in range(len(res["params"])):
        with mlflow.start_run(run_name=f"combo_{i:02d}", nested=True):
            mlflow.log_params({k.replace("xgbclassifier__", ""): v for k, v in res["params"][i].items()})
            mlflow.log_metric("cv_mean_f1", res["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", res["std_test_score"][i])
            mlflow.log_metric("rank", int(res["rank_test_score"][i]))
    print(f"Logged {len(res['params'])} parameter combinations to MLflow")

    # --- best parameters
    best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid.best_params_.items()}
    mlflow.log_params(best_params)
    mlflow.log_params({"scale_pos_weight": round(scale_pos_weight, 3),
                       "threshold": THRESHOLD, "cv_folds": 5, "scoring": "f1"})
    mlflow.log_metric("best_cv_f1", grid.best_score_)
    print("Best params:", best_params)
    print(f"Best CV F1 : {grid.best_score_:.4f}")

    # --- evaluate
    best_model = grid.best_estimator_
    train_prob = best_model.predict_proba(Xtrain)[:, 1]
    test_prob = best_model.predict_proba(Xtest)[:, 1]
    all_metrics = {**metrics(ytrain, train_prob, "train"), **metrics(ytest, test_prob, "test")}
    mlflow.log_metrics(all_metrics)

    test_pred = (test_prob >= THRESHOLD).astype(int)
    print("\nTrain classification report:")
    print(classification_report(ytrain, (train_prob >= THRESHOLD).astype(int)))
    print("Test classification report:")
    print(classification_report(ytest, test_pred))
    print("Test confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(ytest, test_pred))

    # --- save the best model for deployment
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(best_model, model_path)

    meta = {
        "model_file": MODEL_NAME,
        "algorithm": "XGBClassifier (StandardScaler + OneHotEncoder pipeline)",
        "best_params": best_params,
        "best_cv_f1": round(grid.best_score_, 4),
        "threshold": THRESHOLD,
        "metrics": {k: round(v, 4) for k, v in all_metrics.items()},
        "mlflow_run_id": parent.info.run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\nBest model saved to {model_path}")
    print(json.dumps(meta["metrics"], indent=2))
