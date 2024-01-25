# 'dataset' holds the input data for this script

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
    
class ReviewsDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=512):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True
        )
        return {
            'review': review,
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Parameters
batch_size = 16  # Adjust batch size according to your GPU memory
max_length = 256  # Adjust the max_length for the tokenizer


dataset2 = ReviewsDataset(dataset['review'], tokenizer, max_length)
dataloader = DataLoader(dataset2 , batch_size=batch_size)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict_sentiment(data_loader, model):
    model.eval()
    sentiments = []
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            sentiments += predictions.cpu().numpy().tolist()
    return sentiments


dataset['sentiment'] = predict_sentiment(dataloader, model)



