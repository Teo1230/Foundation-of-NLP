
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from utils import *


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
