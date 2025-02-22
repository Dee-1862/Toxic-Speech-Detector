# Initial Libraries.
import sys
import os
from transformers import logging as transformers_logging

# The Tensorflow warnings suppressor.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The HuggingFace warnings suppressor.
transformers_logging.set_verbosity_error()

#Including the parent directory to access utils.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Remaining Libraries.
from utils import seed, file_importer
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Sets seed and imports the processed file.
seed()
final_table = file_importer()

# Example labels and input data.
labels = list(final_table['toxicity'])
posts =list(final_table['posts'])

# Label Encoding.
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Load tokenizer and model.
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('models/DistilBERT/output/pre_trained', num_labels = len(np.unique(encoded_labels)))

# Maximum Length.
token_lengths = []
for token in posts:
    token_lengths.append(len(tokenizer.encode(token)))
max_len = max(token_lengths)

# Tokenization and input formatting.
def tokenize_and_format(texts, labels):
    encodings = tokenizer(texts, padding = True, truncation = True, max_length = max_len, return_tensors = "pt")
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    return dataset

# Setting up for StratifiedKFold.
skf = StratifiedKFold(n_splits = 3, shuffle=True, random_state = 69)
posts_arry = np.array(posts)
lbl_arry = np.array(encoded_labels)

# Iterates over each fold.
for train_indx, val_indx in skf.split(posts_arry, lbl_arry):
    train_texts, valu_texts = posts_arry[train_indx], posts_arry[val_indx]
    train_lbls, valu_lbls = lbl_arry[train_indx], lbl_arry[val_indx]
    train_dataset = tokenize_and_format(train_texts.tolist(), train_lbls.tolist())
    val_dataset = tokenize_and_format(valu_texts.tolist(), valu_lbls.tolist())

# DataLoader setup.
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 32)

# Move model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Learning Rate Scheduler.
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5) # All the learning rate >2e-5 give the same accuracy.
num_epochs = 5 # This is the ideal number of epoches with a correct balance between speed and accuracy.
training_steps = num_epochs * len(train_loader) 
warmup_steps = int(0.06 * training_steps)
lr_scheduler = get_scheduler("linear", optimizer, warmup_steps, training_steps)

# Training loop for the model.
def train(num_epochs):
    model.train()
    total_steps = num_epochs * len(train_loader)

    with tqdm(total = total_steps, desc="Training Progress", unit="batch") as bar:
        for epoch in range(num_epochs):
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                # Forward pass.
                outputs = model(input_ids = input_ids, attention_mask =attention_mask, labels = labels) 
                loss = outputs.loss

                # Backpropogation.
                optimizer.zero_grad()
                loss.backward()

                #Optimization and learning rate scheduler.
                optimizer.step()
                lr_scheduler.step()

                # Updates progress bar.
                bar.set_postfix(epoch =epoch + 1, loss=loss.item())
                bar.update(1)

# Training the model.
train(num_epochs)

# Saves the final DistilBERT model.
model.save_pretrained('models/DistilBERT/output/trained')