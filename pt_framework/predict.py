import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the custom module used in the sequential model.
class LastTimestep(nn.Module):
    def forward(self, inputs):
        # Return the hidden state for the last timestep from the top layer.
        return inputs[1][0][2]

def build_model(encoding_width):
    model = nn.Sequential(
        nn.LSTM(encoding_width, 256, num_layers=3, dropout=0.5, batch_first=True),
        LastTimestep(),
        nn.Dropout(0.5),  # Add this since PyTorch LSTM does not apply dropout to top layer.
        nn.Linear(256, encoding_width)
    )
    return model

def main():
    parser = argparse.ArgumentParser(description='Predict text using a trained model with top-k sampling.')
    parser.add_argument('seed', type=str, help='The seed string to continue from')
    parser.add_argument('num_letters', type=int, help='Number of letters to predict')
    # Optionally, allow the user to set the top-k value.
    parser.add_argument('--top_k', type=int, default=5, help='The number of top probable tokens to consider')
    args = parser.parse_args()
    
    # Load the saved model and metadata.
    checkpoint = torch.load('model.pth', map_location=device)
    char_to_index = checkpoint['char_to_index']
    index_to_char = checkpoint['index_to_char']
    encoding_width = checkpoint['encoding_width']
    
    # Build the model using the same architecture as in training.
    model = build_model(encoding_width)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # For consistency with training, you can use the same window length.
    WINDOW_LENGTH = 40

    # Prepare the initial sequence as a list of one-hot vectors.
    seed = args.seed
    one_hots = []
    for char in seed:
        x = np.zeros(encoding_width, dtype=np.float32)
        # Optionally, handle characters that weren't seen during training.
        if char in char_to_index:
            x[char_to_index[char]] = 1
        one_hots.append(x)
    
    temperature = 0.9  # Adjust this parameter to control randomness.
    top_k = args.top_k
    # Generate new characters one at a time.
    for _ in range(args.num_letters):
        # If the sequence is longer than WINDOW_LENGTH, use only the last WINDOW_LENGTH tokens.
        current_seq = one_hots[-WINDOW_LENGTH:] if len(one_hots) > WINDOW_LENGTH else one_hots
        # Convert to a numpy array with shape (1, sequence_length, encoding_width).
        inputs = torch.from_numpy(np.array(current_seq, dtype=np.float32)[None, :, :]).to(device)
        outputs = model(inputs)
        # Apply temperature and softmax.
        probs = F.softmax(outputs / temperature, dim=1)
        probs_np = probs.cpu().detach().numpy().flatten()
        
        # Get the indices of the top_k probabilities.
        top_k_indices = probs_np.argsort()[-top_k:][::-1]
        top_k_probs = probs_np[top_k_indices]
        # Renormalize the probabilities.
        top_k_probs = top_k_probs / top_k_probs.sum()
        # Sample one index from the top_k indices.
        chosen_index = np.random.choice(top_k_indices, p=top_k_probs)
        # Append the chosen character to the seed.
        chosen_char = index_to_char[chosen_index]
        seed += chosen_char
        
        # Create the one-hot vector for the chosen character and append it.
        one_hot = np.zeros(encoding_width, dtype=np.float32)
        one_hot[chosen_index] = 1
        one_hots.append(one_hot)
    
    # Output the predicted text.
    print(seed)

if __name__ == '__main__':
    main()

