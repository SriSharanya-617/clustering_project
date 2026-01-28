import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🟢",
    layout="wide"
)

# ------------------------------------------------
# LOAD CSS
# ------------------------------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
    """
    <div class="title-box">
        <h1>🟢 Customer Segmentation Dashboard</h1>
        <p>K-Means clustering to group customers based on purchasing behavior</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Clustering Controls")

    numeric_cols = [
        'Fresh', 'Milk', 'Grocery',
        'Frozen', 'Detergents_Paper', 'Delicassen'
    ]

    feature_1 = st.selectbox("Select Feature 1", numeric_cols)
    feature_2 = st.selectbox("Select Feature 2", numeric_cols, index=1)

    k = st.slider("Number of Clusters (K)", 2, 10, 3)

    random_state = st.number_input(
        "Random State",
        min_value=0,
        value=42
    )

    run_btn = st.button("🟦 Run Clustering")

# ------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------
if run_btn:

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # K-MEANS MODEL
    # -----------------------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    st.markdown("## 📊 Customer Clusters")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=df["Cluster"],
        cmap="viridis",
        s=60
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=250,
        marker="X",
        label="Centroids"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("K-Means Customer Segmentation")
    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # CLUSTER SUMMARY
    # -----------------------------
    st.markdown("## 📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")[[feature_1, feature_2]]
        .agg(["mean", "count"])
    )

    st.dataframe(summary)

    # -----------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------
    st.markdown("## 💼 Business Insights")

    for i in range(k):
        st.success(
            f"Cluster {i}: Customers with similar spending patterns in "
            f"{feature_1} and {feature_2}. "
            f"Useful for targeted promotions and inventory planning."
        )

    # -----------------------------
    # USER GUIDANCE
    # -----------------------------
    st.info(
        "📌 Use these clusters to improve marketing strategies, "
        "optimize stock levels, and personalize customer engagement."
    )

else:
    st.info("👈 Select features, choose K, and click **Run Clustering**")
