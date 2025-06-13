# === streamlit_app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import gzip

# Load model artifacts
# with open("model_outputs/best_model.pkl", "rb") as f:
#     best_model = pickle.load(f)
# with open("model_outputs/shap_values.pkl", "rb") as f:
#     shap_values = pickle.load(f)
# with open("model_outputs/label_encoders.pkl", "rb") as f:
#     label_encoders = pickle.load(f)
# with open("model_outputs/feature_columns.pkl", "rb") as f:
#     feature_columns = pickle.load(f)

with gzip.open("model_outputs/best_model.pkl.gz", "rb") as f:
    best_model = pickle.load(f)
with open("model_outputs/shap_values.pkl", "rb") as f:
    shap_values = pickle.load(f)
with open("model_outputs/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("model_outputs/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


X_train = pd.read_csv("model_outputs/X_train.csv")
X_test = pd.read_csv("model_outputs/X_test.csv")
y_test = pd.read_csv("model_outputs/y_test.csv")

label_map = {0: "Poor", 1: "Standard", 2: "Good"}

st.set_page_config(page_title="UMKM Kredit Dashboard", layout="wide")
st.title("📊 Dashboard Kelayakan Kredit UMKM")

# Tabs as Top Navigation
tab_labels = ["📈 Model Comparison", "🧠 Interpretasi Model", "📊 Segmentasi Klaster", "🧮 Prediksi UMKM Baru"]
tabs = st.tabs(tab_labels)

# 1. MODEL COMPARISON
with tabs[0]:
    st.header("📈 Model Comparison")
    df_metrics = pd.read_csv("model_outputs/model_metrics.csv")
    st.dataframe(df_metrics)

    melted = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
    fig = px.bar(melted, x='Metric', y='Value', color='Model', barmode='group', text_auto='.2f',
                 title="Perbandingan Metrik Model")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Confusion Matrix")
    model_selected = st.selectbox("Pilih Model", df_metrics['Model'])
    cm_path = f"model_outputs/conf_matrix_{model_selected.replace(' ', '_')}.npy"
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=list(label_map.values()), y=list(label_map.values()),
                           title=f"Confusion Matrix: {model_selected}")
        st.plotly_chart(fig_cm, use_container_width=True)

# 2. INTERPRETASI MODEL
with tabs[1]:
    st.header("🧠 Interpretasi Model")
    st.subheader("🔎 Global Feature Importance (SHAP)")
    st.image("model_outputs/shap_global.png", caption="Global SHAP Feature Importance", use_container_width=True)

    st.subheader("🔍 Local Explanation")
    index = st.number_input("Pilih Index UMKM (0 - {}):".format(len(X_test)-1), min_value=0, max_value=len(X_test)-1, step=1)
    pred_class = best_model.predict(X_test.iloc[[index]])[0]
    pred_label = label_map.get(pred_class, f"Unknown ({pred_class})")
    st.write("Prediksi:", pred_label)

    exp = shap_values[index]
    single_exp = shap.Explanation(
        values=exp.values[pred_class],
        base_values=exp.base_values[pred_class],
        data=exp.data,
        feature_names=exp.feature_names
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(single_exp, show=False)
    st.pyplot(fig, bbox_inches='tight')

# 3. CLUSTERING
with tabs[2]:
    st.header("📊 Segmentasi UMKM Berdasarkan Klaster")
    df_clusters = pd.read_csv("model_outputs/umkm_clusters.csv")
    fig_cluster = px.scatter(
        df_clusters, x='PCA1', y='PCA2', color=df_clusters['Cluster'].astype(str),
        title="Visualisasi Klaster UMKM (PCA)", labels={'Cluster': 'Cluster'}
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Ringkasan Tiap Klaster")
    df_summary = pd.read_csv("model_outputs/cluster_summary.csv")
    st.dataframe(df_summary)

    st.subheader("Analisis Per Klaster")
    selected_cluster = st.selectbox("Pilih Klaster", sorted(df_clusters['Cluster'].unique()))
    filtered = df_clusters[df_clusters['Cluster'] == selected_cluster]
    st.dataframe(filtered.drop(columns=['PCA1', 'PCA2']))

# 4. PREDIKSI UMKM BARU
with tabs[3]:
    st.header("🧮 Prediksi UMKM Baru")
    st.subheader("Masukkan Informasi UMKM")

    categorical_cols = [
        'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount',
        'Payment_Behaviour', 'UMKM_Kategori'
    ]
    predefined_options = {
        'Type_of_Loan': ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 'Home Equity Loan'],
        'Credit_Mix': ['Good', 'Standard', 'Bad'],
        'Payment_of_Min_Amount': ['Yes', 'No', 'NM'],
        'Payment_Behaviour': [
            'High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
            'Low_spent_Medium_value_payments', 'High_spent_Large_value_payments'
        ],
        'UMKM_Kategori': ['Mikro', 'Kecil', 'Menengah']
    }

    numeric_cols_int = [
        'Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries'
    ]
    numeric_cols_float = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Changed_Credit_Limit',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]

    input_data = {}
    for col in categorical_cols:
        input_data[col] = st.selectbox(col, predefined_options.get(col, []))
    for col in numeric_cols_int:
        input_data[col] = st.number_input(col, value=0, step=1)
    for col in numeric_cols_float:
        input_data[col] = st.number_input(col, value=0.0, step=0.1, format="%.2f")

    income = input_data['Annual_Income']
    debt = input_data['Outstanding_Debt']
    dti = (debt / income * 100) if income > 0 else 0.0
    input_data['DTI'] = round(dti, 3)
    st.markdown(f"📌 **Debt-to-Income (DTI)** dihitung otomatis: `{dti:.3f}%`")

    if st.button("Prediksi Kelayakan"):
        input_df = pd.DataFrame([input_data])
        for col in categorical_cols:
            le = label_encoders.get(col)
            if le:
                input_df[col] = le.transform(input_df[col])
        input_df = input_df[feature_columns]

        pred = best_model.predict(input_df)[0]
        pred_label = label_map.get(pred, f"Unknown ({pred})")
        st.success(f"📊 Prediksi: **{pred_label}**")

        explainer = shap.Explainer(best_model, X_train)
        shap_exp = explainer(input_df)
        exp = shap_exp[0]

        single_exp = shap.Explanation(
            values=exp.values[pred],
            base_values=exp.base_values[pred],
            data=exp.data,
            feature_names=exp.feature_names
        )

        st.subheader("Penjelasan Kenapa (SHAP):")
        fig, ax = plt.subplots()
        shap.plots.waterfall(single_exp, show=False)
        st.pyplot(fig, bbox_inches='tight')
