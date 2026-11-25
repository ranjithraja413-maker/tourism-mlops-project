# Visit with Us - Wellness Tourism Package Prediction (MLOps)

End-to-end MLOps pipeline: data registration -> data prep -> XGBoost tuning with MLflow ->
model committed to `main` -> Streamlit app on Streamlit Community Cloud.

| Path | Purpose |
|---|---|
| `tourism_project/data/` | raw dataset |
| `tourism_project/model_building/` | `data_register.py`, `prep.py`, `train.py` |
| `tourism_project/deployment/` | `app.py`, `requirements.txt`, trained model |
| `.github/workflows/pipeline.yml` | GitHub Actions CI/CD pipeline |
