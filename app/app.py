import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

# map encoding
agency_map = {
    'C2B': 0.059456551, 'LWC': 0.03649635, 'KML': 0.030534351, 'CSR': 0.019607843, 
    'CBH': 0.018867925, 'CCR': 0.012820513, 'TST': 0.005952381, 'JZI': 0.004755614, 
    'RAB': 0.003174603, 'ART': 0, 'CWT': 0, 'ADM': 0
}

product_map = {
    'Silver Plan': 0.067014795, 'Bronze Plan': 0.057259714, 'Single Trip Travel Protect Silver': 0.052631579,
    'Premier Plan': 0.045454545, 'Single Trip Travel Protect Gold': 0.036363636, 'Gold Plan': 0.032520325, 
    'Comprehensive Plan': 0.017045455, 'Value Plan': 0.008141113, 'Travel Cruise Protect': 0.005952381, 
    'Basic Plan': 0.004319033, '24 Protect': 0, 'Rental Vehicle Excess Insurance': 0, 
    'Single Trip Travel Protect Platinum': 0
}

destination_map = {
    'NEW ZEALAND': 0.090909091, 'SINGAPORE': 0.059217578, 'KOREA, REPUBLIC OF': 0.034722222, 
    'VIET NAM': 0.016042781, 'INDONESIA': 0.011538462, 'AUSTRALIA': 0.008849558, 
    'JAPAN': 0.007633588, 'CHINA': 0.006711409, 'MALAYSIA': 0.006163328, 'HONG KONG': 0.00477327, 
    'THAILAND': 0.004296455, 'BRUNEI DARUSSALAM': 0.003164557, 'INDIA': 0, 'PHILIPPINES': 0, 
    'IRAN, ISLAMIC REPUBLIC OF': 0, 'MYANMAR': 0, 'SRI LANKA': 0, 'MACAO': 0, 
    'LAO PEOPLE\'S DEMOCRATIC REPUBLIC': 0, 'TAIWAN, PROVINCE OF CHINA': 0, 'CAMBODIA': 0, 
    'UNITED ARAB EMIRATES': 0, 'BANGLADESH': 0, 'JORDAN': 0, 'MALI': 0, 'IRELAND': 0, 
    'BAHRAIN': 0, 'UNITED KINGDOM': 0, 'SWITZERLAND': 0, 'ANGOLA': 0, 'NEPAL': 0, 'NETHERLANDS': 0, 
    'CANADA': 0, 'UNITED STATES': 0, 'PAKISTAN': 0, 'GUINEA': 0, 'EGYPT': 0, 'PAPUA NEW GUINEA': 0, 
    'ARGENTINA': 0, 'BELGIUM': 0, 'FRANCE': 0, 'ITALY': 0, 'OMAN': 0, 'SWEDEN': 0, 'MONGOLIA': 0
}

# konversi input user ke dalam encoding
def encode_input(agency, product, destination):
    agency_encoded = agency_map.get(agency, 0)
    product_encoded = product_map.get(product, 0)
    destination_encoded = destination_map.get(destination, 0)
    return agency_encoded, product_encoded, destination_encoded

# Load model
model = joblib.load('rf_1.joblib')

# prediksi
def claim_prediction(input_features):
    input_data = pd.DataFrame({
        'Duration': [input_features[0]],
        'Sales': [input_features[1]],
        'Commision': [input_features[2]],
        'Age': [input_features[3]],
        'Type_Travel Agency': [input_features[4]],
        'Distribution_Online': [input_features[5]],
        'Agency_encoded': [input_features[6]],
        'Product_encoded': [input_features[7]],
        'Destination_encoded': [input_features[8]],
    })
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Probabilitas klaim
    return prediction[0], prediction_proba[0]

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 3)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Claim', 'Claim'], yticklabels=['No Claim', 'Claim'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# ROC Curve
def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# SHAP Value
def plot_shap_summary(X, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st.write("### Feature Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:, :, 1], X, show=False)
    st.pyplot(fig)

# metrik utama
def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    metrics_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })
    st.write("### Model Evaluation Metrics")
    st.table(metrics_table)

# evaluasi model
def evaluate_model(y_true, y_pred, y_proba, X):
    
    display_metrics(y_true, y_pred)

    st.write("### Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred)

    st.write("### ROC Curve")
    plot_roc_curve(y_true, y_proba)

    plot_shap_summary(X, model)

# UI
def main():
    st.title('Prediksi Klaim Travel Insurance')
    st.write('Masukkan data dalam format JSON untuk memprediksi klaim.')

    # JSON
    json_input = st.text_area("Paste JSON Data here:")

    if json_input:
        try:
            data = json.loads(json_input)

            y_true = []  # True labels from your actual data (misalnya klaim yang sebenarnya)
            y_pred = []  # Prediksi klaim dari model
            y_proba = []  # Probabilitas prediksi klaim

            features_list = []

            for item in data:
                agency_encoded, product_encoded, destination_encoded = encode_input(item['Agency'], item['Product'], item['Destination'])
                features = [
                    item['Duration'],
                    item['Sales'],
                    item['Commision'],
                    item['Age'],
                    item['Type_Travel Agency'],
                    item['Distribution_Online'],
                    agency_encoded,
                    product_encoded,
                    destination_encoded
                ]
                features_list.append(features)

                # Predik
                prediction, proba = claim_prediction(features)
                y_true.append(item['Claim'])  # Label claim aktual dari JSON
                y_pred.append(prediction)
                y_proba.append(proba)

            # Evaluasi
            X = pd.DataFrame(features_list, columns=[
                'Duration', 'Sales', 'Commision', 'Age', 
                'Type_Travel Agency', 'Distribution_Online', 
                'Agency_encoded', 'Product_encoded', 'Destination_encoded'])

            evaluate_model(y_true, y_pred, y_proba, X)

        except json.JSONDecodeError:
            st.error("Data JSON yang dimasukkan tidak valid. Pastikan formatnya benar.")

if __name__ == '__main__':
    main()
