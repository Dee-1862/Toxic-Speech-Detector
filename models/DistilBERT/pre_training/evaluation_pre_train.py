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
from utils import seed, file_importer, evaluate, TextDataset
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
from torch.utils.data import DataLoader

# Sets seed and imports the processed file.
seed()
final_table = file_importer()

# Loads model.
model = DistilBertForMaskedLM.from_pretrained('models/DistilBERT/output/pre_trained')

# Device dependent (GPU and CPU settings).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Parameters.
posts = final_table['posts']
rationales = final_table['rationales']

# Dataset and DataLoader.
dataset = TextDataset(posts, rationales, tokenizer)
dataloader = DataLoader(dataset, batch_size = 32)

# Evaluates the model.
evaluate(dataloader, device, model, tokenizer)

