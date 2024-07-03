import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Import tqdm for progress bar
from torch.optim import AdamW  # Import PyTorch's AdamW

# Load data from CSV
data_path = '/home/zhanghongyu/panruwei/apkutils-master/apk_info.csv'
data = pd.read_csv(data_path)

# Selecting relevant columns for classification
text_data = data['App Name'].fillna('')  # Replace NaN values with empty strings
features = data[['Package Name', 'Main Activity', 'Activities', 'Services', 'Receivers', 'Permissions', 'MD5', 'Logo Path', 'Cert_SHA1', 'Cert_SHA256', 'Cert_Issuer', 'Cert_Subject', 'Cert_Hash_Algo', 'Cert_Signature_Algo', 'Cert_Serial_Number']].fillna('')  # Replace NaN values with empty strings
labels = data['Label']

# Encode labels to integer
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# BERT Tokenizer
from transformers import BertTokenizer

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
except EnvironmentError:
    print("Tokenizer not found. Downloading...")
    BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models')
    tokenizer = BertTokenizer.from_pretrained('./models/bert-base-uncased', do_lower_case=True)

# Encode text using BERT tokenizer
def preprocess_text_data(text_data, tokenizer, max_len=512):  # 增大 max_len 值，例如 256
    input_ids = []
    attention_masks = []

    for sent in text_data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Encode categorical features
def preprocess_features(features):
    encoded_features = []
    for col in features.columns:
        encoder = LabelEncoder()
        encoded_col = encoder.fit_transform(features[col])
        encoded_features.append(encoded_col)
    encoded_features = np.stack(encoded_features, axis=1)
    return torch.tensor(encoded_features, dtype=torch.float32)

# Preprocess text and additional features
input_ids, attention_masks = preprocess_text_data(text_data, tokenizer)
feature_tensors = preprocess_features(features)
labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long type for classification

# Split data into train and test sets
train_inputs, test_inputs, train_masks, test_masks, train_features, test_features, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, feature_tensors, labels, test_size=0.1, random_state=42
)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader for batching and shuffling
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_features, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_features, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Define BERT Classifier model
class BertClassifier(nn.Module):
    def __init__(self, num_labels=5, num_features=15):  # Assuming 5 classes for labels and 15 additional features
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.feature_layer = nn.Linear(num_features, 128)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 128, num_labels)

    def forward(self, input_ids, attention_mask, features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        feature_output = self.feature_layer(features)
        feature_output = nn.ReLU()(feature_output)
        feature_output = self.dropout(feature_output)

        combined_output = torch.cat((pooled_output, feature_output), dim=1)
        logits = self.classifier(combined_output)

        return logits

# Initialize model, optimizer, scheduler
model = BertClassifier(num_labels=5, num_features=train_features.shape[1])
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # Using PyTorch's AdamW
epochs = 100
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Function to evaluate model
def evaluate(model, dataloader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_features, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, features=b_features)
        logits = outputs
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_steps
    return eval_accuracy

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_features, b_labels = batch

        # Perform forward pass
        outputs = model(b_input_ids, attention_mask=b_input_mask, features=b_features)

        # Debugging: print shapes and values
        # print(f"Logits shape: {outputs.shape}")
        # print(f"Labels shape: {b_labels.shape}")
        # print(f"Logits: {outputs}")
        # print(f"Labels: {b_labels}")

        model.zero_grad()
        loss = loss_fn(outputs, b_labels)
        
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss}")

    # Evaluate at the end of each epoch
    model.eval()
    eval_accuracy = evaluate(model, test_dataloader)
    print(f"Accuracy after epoch {epoch + 1}: {eval_accuracy}")

print("Training complete!")
