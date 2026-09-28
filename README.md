# XAI-NIDS Live Demo

An interactive demo of **XAI-NIDS**, a leakage-controlled and statistically validated explainable
machine-learning framework for network intrusion detection. Enter (or upload) network-flow records,
get an **Attack / Normal** verdict from an XGBoost model, and see **why** through SHAP explanations.

**Live demo:** _add your Streamlit link here_

## What it does

- **Single flow** - edit the main traffic fields (protocol, service, state, duration, packets, bytes,
  TTLs, loads, ...), press *Analyze flow*, and get the probability of attack plus the top-10 features
  driving that decision (SHAP).
- **Batch (CSV)** - upload flows in the original UNSW-NB15 format (up to 2,000 rows). See predictions
  for every row, accuracy if a `label` column is present, a per-row SHAP explanation, and a global
  feature-importance view of the uploaded file.

## Inference pipeline

The app reproduces the training notebook's preprocessing exactly, using artifacts fitted on the
**training set only** (no leakage):

```
raw record
  -> 8 engineered features (byte_ratio clipped at the train 99th percentile)
  -> drop id / attack_cat / label
  -> LabelEncoder for proto / service / state   (unseen category -> -1)
  -> median imputation
  -> StandardScaler
  -> XGBoost  ->  SHAP TreeExplainer
```

The logic lives in [`pipeline.py`](pipeline.py) (no Streamlit dependency); [`app.py`](app.py) is the UI.

## Run locally

```bash
git clone https://github.com/MashrubaTasnim/xai-nids-demo.git
cd xai-nids-demo
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Use **Python 3.12** to match the environment the artifacts were created in. The library versions in
`requirements.txt` are pinned on purpose: pickled scikit-learn / XGBoost / SHAP objects are
version-sensitive, and `artifacts/metadata.json` records the exact versions used.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repository.
2. On <https://share.streamlit.io>, choose the repo, set the main file to `app.py`, and under
   *Advanced settings* select **Python 3.12**.
3. Deploy, then paste the public URL at the top of this README.

## Repository layout

```
app.py              Streamlit UI
pipeline.py         Preprocessing + prediction + SHAP helpers
requirements.txt    Pinned dependencies
artifacts/          Trained model, SHAP explainer, encoders, imputer, scaler, metadata
sample_flows.csv    (optional) 200 labelled flows sampled from the official UNSW-NB15 test set
```

## Limitations

- Research / educational demo, **not** a production intrusion-detection system.
- Trained and evaluated on UNSW-NB15, a public benchmark of lab-generated traffic (2015). Performance
  can drop on traffic from other networks, attack types or time periods; the accompanying research
  discusses this generalization gap.
- Reported performance on the official UNSW-NB15 test partition: accuracy 90.52 %, weighted F1
  90.43 %, ROC-AUC 98.37 %.
- SHAP values explain the model's behaviour, not ground-truth causes of an attack.

## Data and citation

UNSW-NB15: Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network
intrusion detection systems.* Military Communications and Information Systems Conference (MilCIS).

## Author

Mashruba Tasnem Oishi - mashruba.cse@gmail.com
