from utils import *
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import joblib

# Assuming load_text and prep are defined functions
# Step 1: Load the data
txt = load_text()
st = sorted(txt)
texts, labels = [], []

for century in st:
    for data in txt[century]:
        text = prep(data)  # Process text data
        texts.append(text)
        labels.append(century)

# Step 2: Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 3: Save the Label Encoder
joblib.dump(label_encoder, './label_encoder.pkl')  # Save the label encoder

# Step 4: Define a Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }

# Step 5: Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Step 6: Create DataLoader for the entire dataset
dataset = TextDataset(texts, encoded_labels, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Step 7: Define parameter tuning options
learning_rates = [1e-5, 2e-5]  # Example learning rates
epochs_list = [5, 10]  # Number of epochs to try

# Step 8: Training the model with parameter tuning
for lr in learning_rates:
    for epochs in epochs_list:
        print(f"\nTraining with learning rate: {lr} and epochs: {epochs}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0  # Variable to keep track of correct predictions
            total_samples = 0   # Variable to keep track of total samples

            for batch in data_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                # Calculate predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                # Update total correct and total samples
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += predictions.size(0)

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(data_loader)
            accuracy = total_correct / total_samples  # Calculate accuracy
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Step 9: Save the Trained Model and Tokenizer
output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
