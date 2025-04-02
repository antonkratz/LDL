import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the custom module used in the sequential model.
class LastTimestep(nn.Module):
    def forward(self, inputs):
        # Return hidden state for the last timestep.
        return inputs[1][0][2]

def build_model(encoding_width):
    model = nn.Sequential(
        nn.LSTM(encoding_width, 256, num_layers=3, dropout=0.5, batch_first=True),
        LastTimestep(),
        nn.Dropout(0.5), # Add this since PyTorch LSTM does not apply dropout to top layer.
        nn.Linear(256, encoding_width)
    )    
    return model

def main():
    parser = argparse.ArgumentParser(description='Predict text using a trained model.')
    parser.add_argument('seed', type=str, help='The seed string to continue from')
    parser.add_argument('num_letters', type=int, help='Number of letters to predict')
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

    # Prepare the initial beam.
    seed = args.seed
    one_hots = []
    for char in seed:
        x = np.zeros(encoding_width)
        # Optionally, handle characters that weren't seen during training.
        if char in char_to_index:
            x[char_to_index[char]] = 1
        one_hots.append(x)
    beams = [(np.log(1.0), seed, one_hots)]
    BEAM_SIZE = 4
    num_letters = args.num_letters

    # Predict new letters using beam search.
    for i in range(num_letters):
        minibatch_list = [triple[2] for triple in beams]
        minibatch = np.array(minibatch_list, dtype=np.float32)
        inputs = torch.from_numpy(minibatch).to(device)
        outputs = model(inputs)
        temperature = 0.7  # Experiment with this value.
        outputs = F.softmax(outputs / temperature, dim=1)
        y_predict = outputs.cpu().detach().numpy()

        new_beams = []
        for j, softmax_vec in enumerate(y_predict):
            triple = beams[j]
            for k in range(BEAM_SIZE):
                char_index = np.argmax(softmax_vec)
                # Avoid math domain error if softmax probability is zero.
                prob = softmax_vec[char_index] if softmax_vec[char_index] > 0 else 1e-12
                new_prob = triple[0] + np.log(prob)
                new_letters = triple[1] + index_to_char[char_index]
                x = np.zeros(encoding_width)
                x[char_index] = 1
                new_one_hots = triple[2].copy()
                new_one_hots.append(x)
                new_beams.append((new_prob, new_letters, new_one_hots))
                softmax_vec[char_index] = 0  # Zero out to pick the next best option.
        new_beams.sort(key=lambda tup: tup[0], reverse=True)
        beams = new_beams[0:BEAM_SIZE]

    # Output the predicted strings.
    for item in beams:
        print(item[1])

if __name__ == '__main__':
    main()

