import torch
from config import model_paths, tokenizers
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification


def loader(model_name):
    model_path = model_paths.get(model_name)
    tokenizer_type = tokenizers.get(model_name)
    if model_name == "BERT":
        tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
        model = BertForSequenceClassification.from_pretrained(model_path)

    elif model_name == "DistilBERT":
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_type)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

    elif model_name == "HateBERT":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors = 'pt', padding = True, truncation = True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim = 1).cpu().numpy()
    return probabilities