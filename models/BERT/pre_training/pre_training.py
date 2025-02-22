# Importing the initial libraries.
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
from transformers import BertTokenizer, BertForMaskedLM, get_scheduler
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from tqdm import tqdm
from utils import seed, file_importer, TextDataset

seed()
final_table = file_importer()

"""
NOTE: Make sure to change to these values in accordance with your computational resources:
        - batch_size
        - mask_percentage
        - max_length
        - optimizer
        - lr (learning rate)
        - num_epochs (number of epochs)
"""

# Loads tokenizer and model.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncate=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Device dependent GPU and CPU settings.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters
posts = final_table['posts']
rationales = final_table['rationales']

# Dataset and DataLoader.
dataset = TextDataset(posts, rationales, tokenizer)
dataloader = DataLoader(dataset, batch_size = 32, sampler=RandomSampler(dataset))

# Optimizer and learning rate scheduler.
optimizer = AdamW(model.parameters(), lr = 2e-5) 
num_epochs = 5
training_steps = num_epochs * len(dataloader) 
warmup_steps = int(0.06 * training_steps) # Gradual warump: added to prevent instability and divergence.
lr_scheduler = get_scheduler("linear", optimizer = optimizer, num_warmup_steps = warmup_steps, num_training_steps = training_steps)

model.train()
with tqdm(total = training_steps, desc = "Training Progress", unit = "batch") as bar:
    for epoch in range(num_epochs):
        for improve, encode in dataloader:
            improved, encoded = improve.to(device), encode.to(device)
            
            # Forward pass, backpropogation, optimization and learning rate scheduler.
            outputs = model(improved, labels = encoded)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Updates progress bar.
            bar.set_postfix(loss = loss.item())
            bar.update(1)

# Saving the model.
model.save_pretrained('models/BERT/output/pre_trained')

