"""
XAI-NIDS live demo
Explainable network intrusion detection on UNSW-NB15 (XGBoost + SHAP).
Run locally:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

SAMPLE_PATH = Path(__file__).parent / "sample_flows.csv"
MAX_ROWS = 2000
THRESHOLD = 0.5
RED, GREEN = "#C0392B", "#2E8B6E"

st.set_page_config(page_title="XAI-NIDS Demo", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model and preprocessing artifacts...")
def get_artifacts():
    return P.load_artifacts()


try:
    art = get_artifacts()
except Exception as e:  # noqa: BLE001
    st.error(
        "Could not load the model artifacts. This is almost always a library-version mismatch: "
        "the pickles were created with the exact versions listed in `artifacts/metadata.json` "
        "and `requirements.txt` pins them."
    )
    st.exception(e)
    st.stop()

# ----------------------------------------------------------------------------- field metadata
KEY_FIELDS = ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes",
              "rate", "sttl", "dttl", "sload", "dload", "ct_state_ttl"]

HELP = {
    "proto": "Transaction protocol (tcp, udp, icmp, ...).",
    "service": "Network service (http, ftp, ssh, dns, ...). '-' means none / not identified.",
    "state": "State and dependent protocol of the connection (e.g. FIN, CON, INT).",
    "dur": "Total duration of the record, in seconds.",
    "spkts": "Packets from source to destination.",
    "dpkts": "Packets from destination to source.",
    "sbytes": "Bytes from source to destination.",
    "dbytes": "Bytes from destination to source.",
    "rate": "Flow rate, as given in the UNSW-NB15 CSV files.",
    "sttl": "Source-to-destination time to live.",
    "dttl": "Destination-to-source time to live.",
    "sload": "Source bits per second.",
    "dload": "Destination bits per second.",
    "ct_state_ttl": "Connection count per state, by source/destination TTL range.",
}

INT_LIKE = {
    "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "swin", "stcpb",
    "dtcpb", "dwin", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
}

DEFAULTS = P.default_raw_row(art)


def field_widget(name: str):
    """One input widget for one raw feature, pre-filled with the training median."""
    default = DEFAULTS[name]
    help_text = HELP.get(name)
    if name in art.cat_cols:
        options = [str(c) for c in art.encoders[name].classes_]
        return st.selectbox(name, options, index=options.index(str(default)),
                            help=help_text, key=f"in_{name}")
    if name in INT_LIKE:
        return st.number_input(name, value=float(default), step=1.0, format="%.0f",
                               help=help_text, key=f"in_{name}")
    return st.number_input(name, value=float(default), step=0.001, format="%.6f",
                           help=help_text, key=f"in_{name}")


# ----------------------------------------------------------------------------- plots
def plot_contributions(top: pd.DataFrame, title: str):
    top = top.iloc[::-1]  # largest at the top of a horizontal bar chart
    labels = [f"{f} = {v}" for f, v in zip(top["feature"], top["value"])]
    colors = [RED if s > 0 else GREEN for s in top["shap"]]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1.2))
    ax.barh(labels, top["shap"], color=colors)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds)   ←  toward Normal    |    toward Attack  →")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def plot_global(mean_abs: pd.Series, title: str):
    mean_abs = mean_abs.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(mean_abs) + 1.2))
    ax.barh(mean_abs.index, mean_abs.values, color="#5B6C8F")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def show_table(df: pd.DataFrame):
    """Render a DataFrame as plain HTML, avoiding st.table/st.dataframe (both need pyarrow)."""
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)


def explain_one(X_imp_row, sv_row, title):
    display = P.decode_row_for_display(X_imp_row, art)
    top = P.top_contributions(sv_row, display, art, k=10)
    show_fig(plot_contributions(top, title))
    st.caption(
        "Red bars push the prediction toward **Attack**, green bars toward **Normal**. "
        f"Base value: {P.base_value(art):+.2f} log-odds; "
        f"sum of all 50 SHAP values for this flow: {sv_row.sum():+.2f}."
    )
    with st.expander("Show the table"):
        show_table(top.rename(columns={"shap": "SHAP (log-odds)"}))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this demo")
    st.write(
        "A live companion to the **XAI-NIDS** thesis project: a leakage-controlled, "
        "statistically validated, explainable machine-learning framework for network "
        "intrusion detection."
    )
    st.markdown(
        f"**Model:** {art.meta['model_name']}  \n"
        f"**Dataset:** UNSW-NB15 (official train/test partition)  \n"
        f"**Features:** {art.meta['n_features']} "
        f"({len(art.raw_features)} raw + {len(P.ENGINEERED)} engineered)  \n"
        f"**Explainability:** SHAP TreeExplainer"
    )
    st.markdown("**Reported on the official test partition**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "90.52%")
    c2.metric("F1", "90.43%")
    c3.metric("ROC-AUC", "98.37%")
    st.info(
        "Research / educational demo. The model was trained on UNSW-NB15 (lab-generated "
        "traffic from 2015). It is **not** a production intrusion-detection system, and "
        "performance can drop on traffic from other networks or time periods."
    )

# ----------------------------------------------------------------------------- main
st.title("🛡️ XAI-NIDS: Explainable Network Intrusion Detection")
st.write(
    "Score a network flow as **Normal** or **Attack**, and see *why* the model decided that, "
    "feature by feature."
)

tab_single, tab_batch = st.tabs(["Single flow", "Batch (CSV)"])

# ============================================================================= single flow
with tab_single:
    st.write(
        "Fields start at the **training-set median**, i.e. a typical flow. Change any value "
        "and press **Analyze flow**. Real examples are easiest to try in the *Batch (CSV)* tab."
    )
    with st.form("single_form"):
        cols = st.columns(3)
        for i, name in enumerate(KEY_FIELDS):
            with cols[i % 3]:
                field_widget(name)
        other = [f for f in art.raw_features if f not in KEY_FIELDS]
        with st.expander(f"Advanced: {len(other)} more raw features (default = training median)"):
            cols2 = st.columns(3)
            for i, name in enumerate(other):
                with cols2[i % 3]:
                    field_widget(name)
        submitted = st.form_submit_button("Analyze flow")

    if submitted:
        row = {f: st.session_state[f"in_{f}"] for f in art.raw_features}
        X_imp, X_scaled = P.preprocess(pd.DataFrame([row]), art)
        p_attack = float(P.predict_proba_attack(art, X_scaled)[0])
        sv = P.shap_values(art, X_scaled)[0]

        left, right = st.columns([1, 2])
        with left:
            if p_attack >= THRESHOLD:
                st.error("### 🚨 Attack")
            else:
                st.success("### ✅ Normal")
            st.metric("Probability of attack", f"{p_attack:.1%}")
            st.progress(min(max(p_attack, 0.0), 1.0))
            st.caption(f"Decision threshold: {THRESHOLD:.2f}")
        with right:
            explain_one(X_imp[0], sv, "Top 10 features driving this prediction")

# ============================================================================= batch
with tab_batch:
    st.write(
        "Upload a CSV in the **original UNSW-NB15 format** (for example a sample of the official "
        "`UNSW_NB15_testing-set.csv`). Extra columns such as `id`, `attack_cat` and `label` are "
        "allowed; if `label` is present the app also reports accuracy on your file."
    )
    df_in = None
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
    elif SAMPLE_PATH.exists():
        if st.button("Or use the built-in sample flows"):
            st.session_state["use_sample"] = True
        if st.session_state.get("use_sample"):
            df_in = pd.read_csv(SAMPLE_PATH)
            st.caption(f"Using built-in sample: {len(df_in)} flows.")

    if df_in is not None:
        if len(df_in) > MAX_ROWS:
            st.warning(f"File has {len(df_in):,} rows; analyzing the first {MAX_ROWS:,}.")
            df_in = df_in.head(MAX_ROWS)
        try:
            X_imp_b, X_scaled_b = P.preprocess(df_in, art)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        proba = P.predict_proba_attack(art, X_scaled_b)
        sv_b = P.shap_values(art, X_scaled_b)
        pred = np.where(proba >= THRESHOLD, "Attack", "Normal")

        res = pd.DataFrame({"P(attack)": np.round(proba, 4), "Prediction": pred})
        cols_lower = {str(c).strip().lower(): c for c in df_in.columns}
        if "attack_cat" in cols_lower:
            res["Attack category (file)"] = df_in[cols_lower["attack_cat"]].values
        if "label" in cols_lower:
            actual = pd.to_numeric(df_in[cols_lower["label"]], errors="coerce")
            res["Actual"] = np.where(actual == 1, "Attack", np.where(actual == 0, "Normal", "?"))

        m1, m2, m3 = st.columns(3)
        m1.metric("Flows analyzed", f"{len(res):,}")
        m2.metric("Flagged as attack", f"{(pred == 'Attack').mean():.1%}")
        if "Actual" in res:
            known = res["Actual"] != "?"
            if known.any():
                acc = float((res.loc[known, "Actual"] == res.loc[known, "Prediction"]).mean())
                m3.metric("Accuracy on this file", f"{acc:.1%}")

        st.subheader("Predictions")
        show_table(res)

        st.subheader("Explain one flow")
        idx = st.selectbox("Row", list(range(len(res))), format_func=lambda i: f"Row {i}: {pred[i]} ({proba[i]:.1%})")
        explain_one(X_imp_b[idx], sv_b[idx], f"Row {idx}: top features driving this prediction")

        st.subheader("Global view of this file")
        mean_abs = pd.Series(np.abs(sv_b).mean(axis=0), index=art.feature_names)
        show_fig(plot_global(mean_abs, "Most influential features across all uploaded flows"))