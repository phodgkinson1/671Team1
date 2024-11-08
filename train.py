import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load WikiText-103 dataset
print("Loading WikiText-103 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
text_samples = dataset['text']
print(f"Number of text samples: {len(text_samples)}")

# Combine text samples into a single string for tokenization
text = " ".join(text_samples)
print(f"Combined text length: {len(text)} characters")

# Tokenize the text
print("Tokenizing text...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])

# Total unique characters
total_chars = len(tokenizer.word_index) + 1 
print(f"Total unique characters: {total_chars}")

# Convert text to a sequence of integers
print("Converting text to integer sequence...")
input_sequence = tokenizer.texts_to_sequences([text])[0]
print(f"Input sequence length: {len(input_sequence)}")

# Define sequence length
sequence_length = 40
sequences = []
next_chars = []

# Generate sequences for training
print("Generating sequences and next characters for training...")
for i in range(0, len(input_sequence) - sequence_length, 2):  # step size of 2
    sequences.append(input_sequence[i:i + sequence_length])
    next_chars.append(input_sequence[i + sequence_length])
print(f"Total training sequences generated: {len(sequences)}")

# Convert sequences and targets to PyTorch tensors and move to device
X = torch.tensor(sequences, dtype=torch.long).to(device)
y = torch.tensor(next_chars, dtype=torch.long).to(device)
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

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
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
print("DataLoader created.")

# Define the LSTM Model
class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take only the last output for prediction
        x = self.fc(x)
        return x

# Instantiate the model, loss function, and optimizer
embedding_dim = 50
hidden_dim = 256
model = LSTMTextGenerationModel(total_chars, embedding_dim, hidden_dim, total_chars).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model created and initialized.")

# Training loop with debug statements and loss tracking
print("Starting training...")
num_epochs = 10  # Reduced for testing purposes
loss_history = []  # To store loss after each epoch

for epoch in range(num_epochs):
    total_loss = 0
    model.train()  # Ensure model is in training mode
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print loss for every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)  # Store the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}")

print("Training complete.")

# Save the model after training
model_path = "lstm_text_generation_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plotting the loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='b', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()