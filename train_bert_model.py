from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from collections import Counter
import torch
from utils import *


# Load and preprocess the text data
txt = load_text()
txt = {century: [prep(text) for text in texts] for century, texts in txt.items()}

# Step 1: Prepare Data for BERT
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
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Step 2: Convert the century labels to numerical values for classification
century_to_label = {century: idx for idx, century in enumerate(sorted(txt.keys()))}
texts = [text for century_texts in txt.values() for text in century_texts]
labels = [century_to_label[century] for century, century_texts in txt.items() for _ in century_texts]

# Step 3: Augment Data for Minimum Class Representation
def augment_data(texts, labels, min_samples=4):
    label_counts = Counter(labels)
    augmented_texts, augmented_labels = texts.copy(), labels.copy()
    for label, count in label_counts.items():
        if count < min_samples:
            sample_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            for _ in range(min_samples - count):
                augmented_texts.append(texts[sample_indices[0]])
                augmented_labels.append(label)
    return augmented_texts, augmented_labels

augmented_texts, augmented_labels = augment_data(texts, labels, min_samples=6)

# Step 4: Dynamic Train-Test Split
num_classes = len(set(augmented_labels))
test_size = max(0.2, num_classes / len(augmented_labels))  # Ensure at least one sample per class in the test set

train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    augmented_texts, augmented_labels, test_size=test_size, random_state=42, stratify=augmented_labels
)

# Step 5: Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=num_classes
)

# Step 6: Create datasets
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=512)
eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer, max_length=512)

# Step 7: Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 8: Train the Model
trainer.train()

# Step 9: Save the Trained Model and Tokenizer
output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
