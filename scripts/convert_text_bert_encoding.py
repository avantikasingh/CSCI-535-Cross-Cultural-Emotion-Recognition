import csv
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings for a given text
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

# Path to the CSV file
csv_file = 'final_translated.csv'  # Replace with the path to your CSV file

# Open the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file, delimiter=';')

    # Iterate over each row in the CSV file
    for row in reader:
        # Extract the text from the last column (Translated_Text)
        text = row['Translated_Text']
        folder_name = row['Folder']

        # Generate BERT embeddings for the text
        embeddings = generate_bert_embeddings(text)

        # Construct the filename
        filename = "audio_to_text_embeddings.pt"

        # Construct the full path to save the embeddings
        embeddings_file_path = os.path.join('../SEWA/SEWAv02', folder_name, filename)
        os.makedirs(os.path.dirname(embeddings_file_path), exist_ok=True)
        #
        # # Save the embeddings as a .pt file
        torch.save(embeddings, embeddings_file_path)

        print("Embeddings saved:", embeddings_file_path)
