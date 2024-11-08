import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the training corpus
print("Loading WikiText-103 dataset to refit the tokenizer...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
text_samples = dataset['text']
text = " ".join(text_samples)  # Combine samples into a single string
print(f"Loaded text length: {len(text)} characters")

# Initialize and fit the tokenizer on the training text
print("Fitting tokenizer...")
tokenizer = Tokenizer(char_level=True)  # Character-level tokenization
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1  # Total unique characters
print(f"Total unique characters in tokenizer: {total_chars}")

# Define the LSTM model architecture
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

# Model parameters (ensure these match the original training setup)
embedding_dim = 50
hidden_dim = 256
model = LSTMTextGenerationModel(total_chars, embedding_dim, hidden_dim, total_chars)

# Load the model weights
model_path = "lstm_text_generation_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode
print("Model loaded and ready for text generation.")

# Define the text generation function
def generate_text(seed_text, next_chars=100):
    generated_text = seed_text
    for _ in range(next_chars):
        # Tokenize and pad the seed text
        tokenized_input = tokenizer.texts_to_sequences([generated_text])
        tokenized_input = pad_sequences([tokenized_input[0][-40:]], maxlen=40, padding='pre')
        tokenized_input = torch.tensor(tokenized_input, dtype=torch.long).to(device)

        # Predict the next character
        with torch.no_grad():
            output = model(tokenized_input)
            predicted_index = torch.argmax(output, dim=1).item()
        
        # Convert index back to character
        next_char = tokenizer.index_word.get(predicted_index, '')
        generated_text += next_char

    return generated_text

# Generate text using a seed
seed_text = input("Enter seed prompt: ")
generated_output = generate_text(seed_text, next_chars=200)
print("\nGenerated Text:")
print(generated_output)
