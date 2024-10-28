# Required imports
import os
import re
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the Romanian spaCy model for text preprocessing
nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 4000000  # Set a high max_length to avoid truncation issues

# Step 1: Load Text Data by Century
def load_text():
    path = os.getcwd()  # Current working directory
    path_centuries = os.path.join(path, 'century')  # Path to 'century' folder
    txt = {}  # Dictionary to store raw text by century

    for century in os.listdir(path_centuries):
        path_century = os.path.join(path_centuries, century)
        txt[century] = []

        for pdf_file in os.listdir(path_century):
            file_path = os.path.join(path_century, pdf_file)
            if pdf_file.endswith('.txt'):  # Check if the file is a text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text:  # Ensure text is not empty
                    txt[century].append(text)

    return txt

# Step 2: Preprocess Text Data
def has_vowel(word):
    """Check if the word contains at least one vowel."""
    vowels = "aeiouăâî"
    return any(char in vowels for char in word)

def prep(text):
    # Convert text to lowercase and replace specific characters
    text = text.lower()
    text = text.replace("[", "ă").replace("{", "ă").replace("`", "â")
    text = text.replace("=", "ș").replace("\\", "ț").replace("]", "î")
    text = text.replace("}", "Î").replace("|", "Ț").replace("º", "ș")
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs

    # Remove Roman numerals
    roman_numeral_pattern = r'\b(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))\b'
    text = re.sub(roman_numeral_pattern, '', text)

    # Process the text with spaCy
    doc = nlp_ro(text)
    sanitized_tokens = [
        token.lemma_ for token in doc if token.is_alpha and not token.is_stop and
        len(token) > 1 and has_vowel(token.lemma_)
    ]
    sanitized_text = ' '.join(sanitized_tokens)
    return sanitized_text

# Load and preprocess the text data
txt = load_text()
txt = {century: [prep(text) for text in texts] for century, texts in txt.items()}

# Step 3: Prepare Data for BERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Convert the century labels to numerical values for classification
century_to_label = {century: idx for idx, century in enumerate(sorted(txt.keys()))}
texts = [text for century_texts in txt.values() for text in century_texts]
labels = [century_to_label[century] for century, century_texts in txt.items() for _ in century_texts]

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(century_to_label))

# Create dataset and dataloader
dataset = TextDataset(texts, labels, tokenizer, max_length=512)

# Step 4: Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Step 5: Train the Model
trainer.train()

# Step 6: Save the Trained Model and Tokenizer
output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
