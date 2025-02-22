# Initial Libraries.
import sys
import os
from transformers import logging as transformers_logging

# The Tensorflow warnings suppressor.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The HuggingFace warnings suppressor.
transformers_logging.set_verbosity_error()

# Including the parent directory to access utils.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Remaining Libraries.
from utils import seed, file_importer, eval_train
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Sets seed and imports the processed file.
seed()
final_table = file_importer()

# Example labels and input data.
labels = list(final_table['toxicity'])
posts = list(final_table['posts'])

# Label Encoding.
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Load HateBERT tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
model = AutoModelForSequenceClassification.from_pretrained('models/HateBERT/output/trained')

# Maximum Length.
token_lengths = []
for token in posts:
    token_lengths.append(len(tokenizer.encode(token)))
max_len = max(token_lengths)

# Tokenization and input formatting.
def tokenize_and_format(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length = max_len, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    return dataset

# Set up StratifiedKFold.
skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 69)
posts_arry = np.array(posts)
lbl_arry = np.array(encoded_labels)

# Iterate over each fold.
for train_indx, val_indx in skf.split(posts_arry, lbl_arry):
    train_texts, valu_texts = posts_arry[train_indx], posts_arry[val_indx]
    train_lbls, valu_lbls = lbl_arry[train_indx], lbl_arry[val_indx]

    # Tokenize and format the datasets using your predefined function.
    train_dataset = tokenize_and_format(train_texts.tolist(), train_lbls.tolist())
    val_dataset = tokenize_and_format(valu_texts.tolist(), valu_lbls.tolist())

# DataLoader setup.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Move model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Running the evaluation problem.
eval_train(val_loader, model, device)
