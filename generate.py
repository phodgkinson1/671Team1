import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import torch.nn.functional as F

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the WikiText-103 dataset
print("Loading WikiText-103 dataset to refit the tokenizer...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
text_samples = dataset['text'][:10000]
text = " ".join(text_samples)
print(f"Loaded text length: {len(text)} characters")

# Initialize and fit the tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1
print(f"Total unique characters in tokenizer: {total_chars}")

# Define the LSTM model
class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Model parameters
embedding_dim = 50
hidden_dim = 256
model = LSTMTextGenerationModel(total_chars, embedding_dim, hidden_dim, total_chars)

# Load the model
model_path = "lstm_text_generation_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded and ready for text generation.")

# Define the text generation function with temperature scaling
def generate_text(seed_text, next_chars=100, temperature=1.0):
    generated_text = seed_text
    model.eval()
    with torch.no_grad():
        for _ in range(next_chars):
            tokenized_input = tokenizer.texts_to_sequences([generated_text])
            tokenized_input = pad_sequences([tokenized_input[0][-40:]], maxlen=40, padding='pre')
            tokenized_input = torch.tensor(tokenized_input, dtype=torch.long).to(device)
            output = model(tokenized_input)
            probabilities = F.softmax(output / temperature, dim=-1).squeeze()
            next_token_id = torch.multinomial(probabilities, 1).item()
            next_char = tokenizer.index_word.get(next_token_id, '')
            generated_text += next_char
    return generated_text

# Generate text using a seed
seed_text = input("Enter seed prompt: ")
generated_output = generate_text(seed_text, next_chars=200, temperature=0.8)  # Adjust temperature as needed
print("\nGenerated Text:")
print(generated_output)
