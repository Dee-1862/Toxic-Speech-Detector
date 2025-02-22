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
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import DataLoader

seed()
final_table = file_importer()

# Loads model anf tokenizer.
model = BertForMaskedLM.from_pretrained('models/BERT/output/pre_trained')
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# Device dependent (GPU and CPU settings).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters.
posts = final_table['posts']
rationales = final_table['rationales']

# Dataset and DataLoader.
dataset = TextDataset(posts, rationales, tokenizer)
dataloader = DataLoader(dataset, batch_size = 32)

# Evaluates the pre-trained BERT model.
evaluate(dataloader, device, model, tokenizer)

