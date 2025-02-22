from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re 
import unicodedata
import json

########################### preprocessing functions ##########################
# Data importer
def data_importer():
    """
    Purpose: Loads dataset which is a JSON file.

    Input: 
        - None

    Output: 
        - data_df: A dictionary containing the dataset.
    """
    file_path = 'data/raw/dataset.json'
    alternate_path = '../data/raw/dataset.json'
    try:
        with open(file_path, 'r') as file:
            data_df = json.load(file)
    except:
        with open(alternate_path, 'r') as file:
            data_df = json.load(file)
    return data_df

# NOTE: The threshold for both the below helper functions is set to 0.3 to prioritize inclusivity.
# Helper function
def get_majority_label(lbl_list, threshold = 0.3):
    """
    Purpose: Finds the majority label from a list of labels for each post.

    Input: 
        - lbl_list (list): A list of labels.
        - threshold (float): The threshold for considering the label as the majority label.

    Output: 
        - The majority label if (value/total_lbls) > threshold, else None.
    """
    lbl_num = Counter(lbl_list)
    total_lbls = len(lbl_list)
    if total_lbls == 0:
        return None
    else:
        common = lbl_num.most_common(1)
        key, value = common[0]
        if (value/total_lbls) > threshold:
            return key
        else:
            return None
        
# Helper function
def get_majority_targets(annotators, threshold = 0.3):
    """
    Purpose: Finds the majority targets from a list of annotators for each post.

    Input: 
        - annotators (list): A list of entries about the annotators.
        - threshold (float): The threshold for considering the target as the majority target.

    Output: 
        - A list of majority targets and [] if no majority.
    """
    trgt_num = {}
    total_annotations = len(annotators)
    if total_annotations == 0:
        return []
    
    # Counts the occurences of each target
    for occr in annotators:
        for trgt in occr.get('target', []):
            if trgt != "None":
                trgt_num[trgt] = trgt_num.get(trgt, 0) + 1
            else:
                trgt_num["None"] = trgt_num.get("None", 0)

    # Infer the majority target
    majority_targets = []
    for key, value in trgt_num.items():
        if value / total_annotations > threshold:
            majority_targets.append(key)

    if not majority_targets:
        return ["None"]
    return majority_targets

def clean_emojis(text):
    """
    Purpose: Removes emojis and normalizes text.

    Input: 
        - text (str): The input text containing emojis.

    Output: 
        - text (str): The cleaned text without emojis.
    """
    emoji_pattern = re.compile("["
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001FA00-\U0001FA6F"  # Chess pieces and symbols
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U0001F780-\U0001F7FF"  # Geometrical shapes
        "\U0001F300-\U0001F5FF"  # Miscellaneous symbols and pictographs
        "\U0001FA70-\U0001FAFF"  # More symbols
        "\U0001F800-\U0001F8FF"  # Supplemental symbols (I)
        "\U0001F900-\U0001F9FF"  # Supplemental symbols (II)
        "\U0001F680-\U0001F6FF"  # Transport symbols
    "]+", flags=re.UNICODE)

    text = unicodedata.normalize("NFKC", text)
    text = emoji_pattern.sub("", text)
    return text



########################## EDA functions ##########################
# Processed Data importer
def final_importer():
    """
    Purpose: Loads processed dataset which is a JSON file.

    Input: 
        - None

    Output: 
        - final_df: A dictionary containing the processed dataset.
    """

    file_path = 'data/processed/final_dataset.json'
    alternate_path = '../data/processed/final_dataset.json'
    try:
        with open(file_path, 'r') as file:
            final_df = json.load(file)
    except:
        with open(alternate_path, 'r') as file:
            final_df = json.load(file)
    return final_df



# Missing counter
def empty_lists(data, column_name):
    """
    Purpose: Finding number of empty lists in a specified column of the dataframe.

    Inputs: 
        - data (pd.DataFrame): A dataFrame containing the dataset.
        - column_name (str): The name of the column in which you want to check for empty lists.

    Output: 
        - misssing_count (int): A sum of the rows with missing values in a specificed column.
    """
    missing_count = 0
    for value in data[column_name]:
        if len(value) == 0:
            missing_count += 1
    return missing_count


# Helper function
def missing_annotators(annotators):
    """
    Purpose: Finds the number of missing targets and labels from a list of annotators.

    Input: 
        - annotators (list): A list of entries about the annotators.

    Output: 
        - missing (int): The number of annotators with missing targets and labels.
    """
    # Base cases
    missing = 0
    target_missing = True
    lable_missing = True
    total_annotations = len(annotators)
    
    if total_annotations == 0:
        return missing

    for annotations in annotators:
        targets = annotations.get('target', [])
        label = annotations.get('label', "")

        if isinstance(targets, list):
            for target in targets:
                if target in [None, "", 'None']:
                    target_missing = True
                else:
                    target_missing = False

        if label in [None, '', 'None']:
            lable_missing = True
        else:
            lable_missing = False
        
        if target_missing or lable_missing:
            missing += 1
   
    return missing



def annotator_counter(data):
    """
    Purpose: Finds the number of missing targets and labels from the annotators column of the dataframe.

    Input: 
        - data (pd.DataFrame): A dataFrame containing the dataset.

    Output: 
        - missing_num (int): The number of rows with missing targets and/or labels.
    """
    missing_num = 0
    for annotations in data['annotations']:
        if missing_annotators(annotations) >= 3:
            missing_num += 1
        else:
            missing_num += 0
    return missing_num



# Helper function
def missing_targets(annotators):
    """
    Purpose: Finds the number of missing targets from a list of annotators.

    Input: 
        - annotators (list): A list of entries about the annotators.

    Output: 
        - trgt_missing (int): The number of annotators with missing targets.
    """
    trgt_missing = 0
    total_annotations = len(annotators)
    
    if total_annotations == 0:
        return trgt_missing

    for annotations in annotators:
        targets = annotations.get('target', [])
        for target in targets:
            if target in [None, "", 'None']:
                trgt_missing += 1
            else:
                trgt_missing += 0
   
    return trgt_missing

def target_counter(data):
    """
    Purpose: Finds the number of missing targets and labels from the annotators column of the dataframe.

    Input: 
        - data (pd.DataFrame): A dataFrame containing the dataset.

    Output: 
        - missing_num (int): The number of rows with missing targets and/or labels.
    """
    missing_num = 0
    for annotations in data['annotations']:
        if missing_targets(annotations) >= 3:
            missing_num += 1
        else:
            missing_num += 0
    return missing_num



# Helper function
def missing_labels(annotators):
    """
    Purpose: Finds the number of missing labels from a list of annotators.

    Input:
        - annotators (list): A list of entries about the annotators.

    Output: 
        - label_missing (int): The number of annotators with missing labels.
    """
    label_missing = 0
    total_annotations = len(annotators)
    
    if total_annotations == 0:
        return label_missing
    
    for annotations in annotators:
        label = annotations.get('label', '')
        if label in [None, '', 'None']:
            lable_missing = True
        else:
            lable_missing = False
        
        if lable_missing:
            label_missing += 1
   
    return label_missing

def label_counter(data):
    """
    Purpose: Finds the number of missing labels from the annotators column of the dataframe.

    Input: 
        - data (pd.DataFrame): A dataFrame containing the dataset.

    Output: 
        - missing_num (int): The number of rows with missing labels.
    """
    missing_num = 0
    for annotations in data['annotations']:
        if missing_labels(annotations) >= 3:
            missing_num += 1
        else:
            missing_num += 0
    return missing_num



def pie_plot(counts, title):
    """
    Purpose: Plots a pie chart for the given counts.

    Input: 
        - counts (pd.Series): A series containing the counts.
        - title (str): The title of the pie chart.

    Output: 
        - Pie plot.
    """
    titles = counts.index
    plt.subplots(figsize = (12,9))
    plt.pie(counts, autopct='%1.2f%%', startangle = 197)
    plt.axis('scaled')
    plt.title(title)
    plt.legend(titles, title="Labels", loc="lower right")
    plt.show()


def word_cloud(words, randomstate):
    """
    Purpose: Plots a word cloud for the given words.

    Input: 
        - counts (pd.Series): A series containing the words.
        - randomstate (int): The seed for the random number generator of the word cloud plot.

    Output: 
        - Word cloud plot.
    """
    text = ' '.join(words)
    wordcloud = WordCloud(width = 900, height = 600, background_color='white', random_state = randomstate)
    finalcloud = wordcloud.generate(text)
    plt.figure(figsize=(12, 9))
    plt.imshow(finalcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def bar_plot(counts, title, xlabel):
    """
    Purpose: Plots a bar plot for the given counts.

    Input: 
        - counts (pd.Series): A series containing the counts.
        - title (str): The title of the bar chart.
        - xlabel(str): The title for the xaxis of the bar chart. 

    Output: 
        - Bar plot.
    """
    titles = counts.index
    plt.figure(figsize=(12, 9))
    plt.bar(titles, counts, alpha = 0.7)
    plt.title(title, fontsize = 23, pad = 20)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.xticks(rotation = 45)
    plt.show()

