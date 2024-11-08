import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import re
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load a subset of WikiText-103 dataset
print("Loading a subset of WikiText-103 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:5%]")  # Reduced dataset size
text_samples = dataset['text']
text = " ".join(text_samples)

# Filter out unwanted symbols
print("Filtering text...")
filtered_text = re.sub(r'[^A-Za-z0-9.,\'\s-]', '', text)
print(f"Filtered text length: {len(filtered_text)} characters")

# Tokenize text
print("Tokenizing text...")
tokenizer = Tokenizer()  # Word-level tokenizer
tokenizer.fit_on_texts([filtered_text])
total_words = len(tokenizer.word_index) + 1
print(f"Total unique words: {total_words}")

# Convert text to a sequence of integers
print("Converting text to integer sequence...")
input_sequence = tokenizer.texts_to_sequences([filtered_text])[0]

# Define sequence length and steps per epoch
sequence_length = 40  # Reduced sequence length
steps_per_epoch = 1
sequences = []
next_words = []

# Generate sequences for training
print("Generating sequences and next words for training...")
for i in range(0, len(input_sequence) - sequence_length, steps_per_epoch):
    sequences.append(input_sequence[i:i + sequence_length])
    next_words.append(input_sequence[i + sequence_length])
sequences = pad_sequences(sequences, maxlen=sequence_length, padding='pre')

# Convert sequences and targets to PyTorch tensors and move to device
X = torch.tensor(sequences, dtype=torch.long).to(device)
y = torch.tensor(next_words, dtype=torch.long).to(device)

# Define PyTorch Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)  # Reduced batch size

# Define the LSTM Model
class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=768, output_dim=None):
        super(LSTMTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Account for bidirectionality

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take only the last output for prediction
        x = self.fc(x)
        return x

# Instantiate the model, loss function, and optimizer
embedding_dim = 200
hidden_dim = 768
model = LSTMTextGenerationModel(total_words, embedding_dim, hidden_dim, total_words).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)  # Updated scheduler parameters

# Training loop with gradient accumulation
print("Starting training...")
num_epochs = 10  # Reduced epochs for initial testing
accumulation_steps = 2  # Number of mini-batches to accumulate gradients

loss_history = []
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accumulation_steps  # Normalize loss by accumulation steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Apply gradient clipping
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up for tracking
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item() * accumulation_steps:.4f}")
        
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}")

print("Training complete.")
model_path = "lstm_word_generation_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save tokenizer for use in generation
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to tokenizer.pkl")

# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()