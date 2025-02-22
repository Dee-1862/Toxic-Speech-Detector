# The warnings suppressor
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

import pandas as pd
import streamlit as st
import numpy as np
from config import model_paths
from utils import loader, predict
import openpyxl

st.title("Batch Analysis of Toxic Speech")

# Load tokenizer and model
selected_model = st.selectbox("Select a Model", model_paths.keys())
tokenizer, model = loader(selected_model)

# Streamlit features
upl_file = st.file_uploader("Upload File", type=["xlsx", "csv"], accept_multiple_files = False)
out_type = st.multiselect("Select Output Type", ["Toxicity (binary)", "Toxicity (label)", "Confidence Score"], [])

if upl_file is not None:
    if upl_file.name.endswith('.csv'):
        df = pd.read_csv(upl_file)
    else:
        df = pd.read_excel(upl_file, engine = 'openpyxl')

    # Condition: dataframe has the required columns
    if 'Text' not in df.columns:
        st.error("File must contain 'text' column.")
    else:
        batch_texts = df['Text'].tolist()
        batch_probabilities = predict(model, tokenizer, batch_texts)

        results = []
        for idx, text in enumerate(batch_texts):
            probabilities = batch_probabilities[idx]
            predicted_class = np.argmax(probabilities)
            predicted_prob = probabilities[predicted_class] * 100

            result = {"Input": text}

            if "Toxicity (binary)" in out_type:
                if predicted_class == 1:
                    result["toxicity"] = 1
                else:
                    result["toxicity"] = 0

            if "Toxicity (label)" in out_type:
                if predicted_class == 1:
                    result["toxicity_label"] = "Toxic"
                else:
                    result["toxicity_label"] = "Non-Toxic"

            if "Confidence Score" in out_type:
                result["confidence"] = f"{predicted_prob:.3f}%"
            
            results.append(result)
        
        result_df = pd.DataFrame(results)
        st.download_button("Download Results", result_df.to_csv(index = False), "results.csv", "text/csv")
        st.write(result_df)
