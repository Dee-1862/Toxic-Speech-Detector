# The warnings suppressor
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)


import streamlit as st
import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import time
from config import model_paths
from utils import loader, predict


st.title("Toxic Speech Detector")

selected_model = st.selectbox("Select a Model:", model_paths.keys())
tokenizer, model = loader(selected_model)

text_input = st.text_area(label = "Input text for analysis:", placeholder = "Enter your text here")

if st.button("Break it down!"):
    if text_input.strip():
        start_t = time.time()
        probs = predict(model, tokenizer, [text_input])[0]
        pred_class = np.argmax(probs)
        predicted_prob = probs[pred_class] * 100

        if pred_class == 1:
            prediction = "Toxic Speech Detected"
        else:
            prediction = "Non-Toxic Speech Detected"

        # Exaplainable AI
        explainer = LimeTextExplainer(class_names = ['Non-Toxic', 'Toxic'])
        explanation = explainer.explain_instance(text_input, lambda text: predict(model, tokenizer, text))

        word_scores = {}
        for feature, importance in explanation.as_list():
            word_scores[feature] = importance
        
        top_words = sorted(
            word_scores.items(),
            key = lambda scrs: abs(scrs[1]),
            reverse = True
        )[:10]
        top_word_scores = dict(top_words)

        max_value = None
        for imp in word_scores.values():
            if max_value is None or imp > max_value:
                max_value = imp

        # Maximum threshold calculation
        max_threshold = 0.60 * max_value

        # Toxicity highlighter
        highlighted_wrds = []
        highlighted_txt = ""
        words = text_input.split()
        for word in words:
            stripped_word = word.strip('?.!,"')
            importance = word_scores.get(stripped_word, 0)

            if importance >= max_threshold:
                highlighted_txt += f' <span style = "background-color: red;">{word}</span>'
                highlighted_wrds.append(stripped_word)
            else:
                highlighted_txt += f' {word}'
        
        # Results
        st.write("### Top toxic words are highlighted:")
        st.markdown(highlighted_txt, unsafe_allow_html = True)
        st.write(f"**Prediction:** {prediction} ({predicted_prob:.2f}% confidence)")

        # Bar plot
        words_list = list(top_word_scores.keys())
        importance_values = list(top_word_scores.values())
        colors = []
        for word, imp in zip(words_list, importance_values):
            if word in highlighted_wrds:
                colors.append("darkred")
            elif imp > 0:
                colors.append("lightcoral")
            else:
                colors.append("green")
        fig, axs = plt.subplots(figsize = (13, 9))
        axs.barh(words_list, importance_values, color = colors, edgecolor = 'black')
        axs.axvline(0, color = 'black', linewidth = 1)
        axs.set_xlim(-1, 1)
        axs.set_xlabel("Importance")
        axs.set_title("Top Importance Words")
        sns.despine()
        st.pyplot(fig)


        end_t = time.time()
        elapsed_t = end_t - start_t
        st.write(f"**Time Elapsed:** {elapsed_t:.3f} seconds")
    else:
        st.warning("Please enter text for analysis.")
