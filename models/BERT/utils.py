import json
import pandas as pd
from transformers import set_seed
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score

# Set seeds.
def seed():
    seed = 69
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

# Importing processed data.
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
    final_table = pd.DataFrame(data = final)
    return final_table

# Evaluation of the trained model.
def evaluate(dataloader, device, model, tokenizer):
    model.eval()
    batch_size = 0
    total_accur = 0

    for input, label in dataloader:
        inputs, labels = input.to(device), label.to(device)
        with torch.no_grad():
            # Get the model outputs (logits) for the input data.
            outputs = model(inputs, labels = labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim = -1)

            # Create a mask to exclude special tokens from the calculation .       
            mask = (inputs != tokenizer.cls_token_id) & (labels != -100) & (inputs != tokenizer.sep_token_id)
            correct = (predictions == labels) & mask
            if mask.sum().item() > 0:
                accuracy = correct.sum().item() / mask.sum().item()  
            else:
                0
            batch_size += 1
            total_accur += accuracy

    fin_accur = total_accur / batch_size
    print(f"Average Accuracy: {fin_accur}")


# TextDataset Class.
class TextDataset(Dataset):
    def __init__(self, posts, rationales, tokenizer):
        self.tokenizer = tokenizer
        self.posts = posts
        self.rationales = rationales
        token_lengths = []
        for token in self.posts:
            token_lengths.append(len(self.tokenizer.encode(token)))
        self.max_len = max(token_lengths)

    # Returns the length of the dataset which is used by the dataloader.
    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        posts = self.posts[idx]
        rationale = self.rationales[idx]

        # Tokenizes, encodes, and includes special tokens ([CLS], [SEP]).
        tokens = self.tokenizer.tokenize(posts)
        encoded_tkns = self.tokenizer.encode(tokens, add_special_tokens=True)

        # Rationes are given to the tokenized words (mainly the exceptions).
        new_rationale = []
        words = posts.split()
        for i in range(len(words)):
            word = words[i]
            rat = rationale[i]
            subwords = self.tokenizer.tokenize(word)
            subword_rat = [rat] * len(subwords)
            new_rationale.extend(subword_rat)
        
        # Adds rationales for the special tokens.
        new_rationale = [0] + new_rationale + [0]

        # Masking .
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

        # Padding.
        padding_length = self.max_len - len(improved_tkns)
        if padding_length > 0:
            improved_tkns.extend([self.tokenizer.pad_token_id] * padding_length)
            encoded_tkns.extend([-100] * padding_length)

        return torch.tensor(improved_tkns), torch.tensor(encoded_tkns)

# Evaluation of the final trned model
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