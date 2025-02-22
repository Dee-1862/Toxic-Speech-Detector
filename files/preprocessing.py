from utils import data_importer, get_majority_label, get_majority_targets, clean_emojis
import pandas as pd
import numpy as np
import re
import json

data_df = data_importer()

# Filtering out missing data
data_df.pop('24439295_gab', None)  # Removes the key if it exists, does nothing otherwise

# Process rationales and keep only valid rows.
rationales = []
valid_rows = []

for key in data_df.keys(): 
    rational = data_df[key]['rationales']
    tokens = data_df[key]['post_tokens']
    
    if rational is None:
        rationales_array = np.array([])
    else:
        rationales_array = np.array(rational)

    if rationales_array.size > 0:
        finalized_rationales = []
        for value in rationales_array.flatten().tolist():
            if int(value) == 1:
                finalized_rationales.append(1)
            else:
                finalized_rationales.append(0)
    else:
        finalized_rationales = []

    # Retrieve label and target information before finalizing rationales
    annotators = data_df[key]['annotators']
    labels = []
    for item in annotators:
        labels.append(item['label'])
    assigned_label = get_majority_label(labels)
    target_value = get_majority_targets(annotators)

    # Replaces empty rationales with a list of zeros, if label is 'normal', target is ['None'], and finalized rationales is an empty list.
    if assigned_label == 'normal' and target_value == ['None'] and not finalized_rationales:
        finalized_rationales = [0] * len(tokens)

    # Convert numpy arrays to lists before storing
    if isinstance(finalized_rationales, np.ndarray):
        finalized_rationales = finalized_rationales.tolist()

    # Retain only rows where rationales exist
    if finalized_rationales:
        rationales.append(finalized_rationales)
        valid_rows.append(key)

# Process post IDs.
post_ids = []
for key in valid_rows:
    post_ids.append(data_df[key]['post_id'])

# Process labels.
label_list = []
for key in valid_rows:
    annotators = data_df[key]['annotators']
    labels = []
    for item in annotators:
        labels.append(item['label'])
    assigned_label = get_majority_label(labels)
    label_list.append(assigned_label)

# Transform labels into binary toxicity categories.
transform_lbl = []
for lbl in label_list:
    if lbl in ['hatespeech', 'offensive']:
        transform_lbl.append('toxic')
    else:
        transform_lbl.append('non-toxic')

# Process targets.
target = []
for key in valid_rows:
    assigned_targets = get_majority_targets(data_df[key]['annotators'])
    target.append(assigned_targets)

# Process post text.
texts = []
for key in valid_rows:
    tokens = data_df[key]['post_tokens']
    texts.append(' '.join(tokens))

# Create DataFrame.
df = pd.DataFrame({
    'post_id': post_ids,
    'label': label_list,
    'toxicity': transform_lbl,
    'target': target,
    'post': texts,
    'rationales': rationales
})

# Clean HTML tags from 'post' while keeping corresponding rationales
for index, row in df.iterrows():
    tokens = row['post'].split()
    rationales = row['rationales']
    # Remove HTML tags and retain corresponding rationales
    cleaned_tokens = []
    cleaned_rationales = []
    for token, rationale in zip(tokens, rationales):
        if not re.match(r'<.*?>', token):  # Remove HTML tags
            cleaned_token = clean_emojis(token) # Normalizes and removes the emojis
            cleaned_tokens.append(cleaned_token)
            cleaned_rationales.append(rationale)

    # Update DataFrame with cleaned values
    df.at[index, 'post'] = ' '.join(cleaned_tokens)
    df.at[index, 'rationales'] = cleaned_rationales

# Define the final target list (targets with >500 related posts).
final_target_list = ['African', 'Islam', 'Women', 'Jewish', 'Homosexual', 'Refugee', 'Arab', 'Caucasian', 'Men', 'Asian', 'Hispanic']

# Filter and retain only valid targets.
for index, row in df.iterrows():
    updated_targets = []
    for trgt in final_target_list:
        if trgt in row['target']:
            updated_targets.append(trgt)

    # If no common target is found, target column is assigned 'None'.
    if not updated_targets:
        df.at[index, 'target'] = ['None']
    else:
        df.at[index, 'target'] = updated_targets

# Save the dataset as a JSON file.
final_df = df.to_dict(orient='records')
with open("data/processed/final_dataset.json", "w") as file:
    json.dump(final_df, file, indent=3)
