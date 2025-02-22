import json
import pandas as pd
from transformers import set_seed
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score

# Sets seed
def seed():
    seed = 69
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

# Imports the processed data
def file_importer():
    file_path = 'data/processed/final_dataset.json'
    with open(file_path, 'r') as file:
        final_df = json.load(file)

    post_id = []
    label = []
    toxicity = []
    target = []
    rationales = []
    post = []

    for value in final_df:
        post_id.append(value['post_id'])
        label.append(value['label'])
        toxicity.append(value['toxicity'])
        target.append(value['target'])
        rationales.append(value['rationales'])
        post.append(value['post'])

    final = {
        'post_id': post_id,
        'label': label,
        'toxicity': toxicity,
        'target': target,
        'rationales': rationales,
        'posts': post
    }

    fin_table = pd.DataFrame(data = final)
    return fin_table

# TextDataset Class
class TextDataset(Dataset):
    # Initializes the TextDataset class.
    def __init__(self, posts, rationales, tokenizer):
        self.tokenizer = tokenizer
        self.posts = posts
        self.rationales = rationales
        token_lengths = []
        for token in self.posts:
            token_lengths.append(len(self.tokenizer.encode(token)))
        self.max_len = max(token_lengths)
    
    # Returns the length of the dataset.
    def __len__(self):
        return len(self.posts)

    # Gets an item from the dataset at the given index.
    def __getitem__(self, idx):
        posts = self.posts[idx]
        rationale = self.rationales[idx]

        tokens = self.tokenizer.tokenize(posts)
        encoded_tkns = self.tokenizer.encode(tokens, add_special_tokens = True)

        new_rationale = []
        words = posts.split()
        for i in range(len(words)):
            word = words[i]
            rat = rationale[i]
            subwords = self.tokenizer.tokenize(word)
            subword_rat = [rat] * len(subwords)
            new_rationale.extend(subword_rat)
        
        new_rationale = [0] + new_rationale + [0]

        position = []
        for indx, rat in enumerate(new_rationale):
            if rat != 0:
                position.append(indx)
        mask_percent = 0.15
        maskables = int(len(position) * mask_percent)
        fin_masks = np.random.choice(position, maskables, replace = False)
        improved_tkns = encoded_tkns[:]

        for indx in fin_masks:
            improved_tkns[indx] = self.tokenizer.mask_token_id

        padding_length = self.max_len - len(improved_tkns)
        if padding_length > 0:
            improved_tkns.extend([self.tokenizer.pad_token_id] * padding_length)
            encoded_tkns.extend([-100] * padding_length)
        
        tensor_impr_tkns = torch.tensor(improved_tkns)
        tensor_encd_tkns = torch.tensor(encoded_tkns)

        return tensor_impr_tkns, tensor_encd_tkns

# Evaluates a trained model
def eval_train(loader, model, device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim = 1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_values.extend(labels.cpu().numpy())


    f1 = f1_score(true_values, predictions, average = 'weighted')
    accuracy = accuracy_score(true_values, predictions)
    print(f"F1-Score (Weighted): {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
