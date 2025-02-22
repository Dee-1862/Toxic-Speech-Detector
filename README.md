# Toxicity-Speech-Detector:

The **Toxicity Speech Detector** is a machine learning model developed using BERT models to classify toxic language texts. This project identifies and classifies toxic speech models using pre-trained transformer models like BERT, DistilBERT and HateBERT. The primary aim of this project is to create an explainable and efficient application that classifies toxic speech speech from texts collected from online forums and social media platforms.

## Table of Contents:

1. [Overview](#overview)
2. [Background](#background)
3. [Dataset & Preprocessing](#dataset-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Streamlit](#streamlit)
7. [References](#references)

## Overview:

The **Toxicity Speech Detector** is a project with multiple pre-trained transformer models such as BERT, DistilBERT, and HateBERT. This project uses an efficient pipeline of pre-analysis, pre-processing, analysis, pre-training, training, evaluation and deployment of the models for toxicity detection. Moreover, this project is developed for analyzing texts (both single texts and large datasets) and detecting toxic speech in a variety of contexts.

## Background:

**Toxic speech (including hate speech and offensive speech)** has become a growing obstacle across all online forums and social media platforms, ranging from Twitter to Reddit. Although the current models for detecting toxic languages are continuously improving, there is still room for advancement in terms of model explainability, efficiency, and transparency. The main goal of this project is to bridge a gap by providing an efficient solution for toxicity detection using NLP, transformer models, and explainable AI.

## Dataset & Preprocessing:

- The dataset used in this project is the **HateXplain Dataset**, a thorough collection for detecting toxic language (including hate speech, offensive speech, and non-toxic speech). Moreover, this dataset plays a crucial role in increasing the efficiency of the model as it includes target communities assigned by each annotator, labels assigned by each annotator, and rationales for each word in the post.
- The data pre-processing involves cleaning the dataset, tokenization, and structuring the raw dataset to make it compatible for training the models. The data is analyzed both before and after the pre-processing to ensure that the text and its corresponding annotations are properly structured.
- The preprocessed dataset is stored in the data/processed/final_dataset.json.

## Model Training:
To train the model on custom datasets, we use the following steps:
1. **Set up:**
   - The models are structed and placed according to their respective names.
   - The reusable code is placed in the 'utils.py' as per requirements (separate for each model).

2. **Pre-training:**
   - First ran the pre-training process for the BERT and DistilBERT models and then moved to the training and evaluation.
   - The pre-trained model is stored in:
     ```
     models/<model-name>/output/pre_trained/
     ```
3. **Training:**
   - After pre-training all the models are trained except for the "HateBERT Model" which is directly trained.
   - The trained model is stored in:
     ```
     models/<model-name>/output/trained/
     ```

## Evaluation:
- The performance of both the pre-trained and the trained models is evaluated using various metrics such as accuracy and F1 score.
- After pre-training the BERT and DistilBERT models, run this evaluation script:
  ```
  models/<model-name>/pre_training/evaluation_pre_train.py
  ```
- After training all the models, run this evaluation script:
  ```
  models/<model-name>/training/evaluation_train.py
  ```
- The **DistilBERT model** achieved the highest performance, with an Accuracy of 87.2% and F1-score of 86.9%.

## Streamlit:
To start the Streamlit dashboard, run the following command:
```
streamlit run app/app.py --server.fileWatcherType=none
```

#### Dashboard Features:
  1. Single Text Analysis:
     - Allows the users to select from a list of models (BERT, DistilBERT, HateBERT).
     - Allows users to input a single piece of text for rapid analysis.
     - The dashboard processes the input text using the selected model and displays the following output:
     - Highlighted toxic words:
     - Prediction with confidence score:
     - Plot: A horizontal bar plot to visualize the importance of each word in the input text. Moreover, here is the meaning associated with the color coding:
       - Dark red: Highlighted words that are considered to be the highly toxic words in the input.
       - Light coral: Non-highlighted words with a positive influence on the prediction.
       - Green: Words with negative influence on the predictions.
     - Screenshots:
       - Single Text Analysis (Before Processing):
         ![Single Text Input](/screenshots/single_text_analysis/before.png)
       - Single Text Analysis (Available Models):
         ![Single Text Input](/screenshots/single_text_analysis/models.png)
       - Single Text Analysis (After Processing):
         ![Toxic Speech Explanation](/screenshots/single_text_analysis/after.png)

  2. Batch Analysis:
     - Allows the users to select from a list of models (BERT, DistilBERT, HateBERT).
     - Allows users to upload a file containing multiple input text in a file for bulk processing.
     - It also allows the users to select output type from a multi-select (containing "Toxicity (binary)", "Toxicity (label)" and "Confidence Score").
     - The dashboard processes the input text file using the selected model and displays the following output based on the input for the multi-select.
     - The final results can be seen in the dashboard and downloaded as a CSV file.
     - Screenshots:
       - Batch Analysis (Before Processing):
         ![Single Text Input](/screenshots/batch_analysis/before.png)
       - Batch Analysis (After Processing):
         ![Toxic Speech Explanation](/screenshots/batch_analysis/after.png)

## References:
```
@inproceedings{mathew2021hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}
```
