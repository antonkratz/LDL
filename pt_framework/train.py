import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = '../data/pg10662.txt'

# WINDOW_LENGTH and WINDOW_STEP are used to control the process of splitting up this text file into multiple training examples
WINDOW_LENGTH = 40
WINDOW_STEP = 3

# Open the input file.
with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as file:
    text = file.read()

# Make lower-case and remove newline and extra spaces.
text = text.lower().replace('\n', ' ').replace('  ', ' ')

# Encode characters as indices.
unique_chars = list(set(text))
char_to_index = {ch: idx for idx, ch in enumerate(unique_chars)}
index_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}
encoding_width = len(char_to_index)

# Create training examples.
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

# Convert to one-hot encoded training data.
X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width), dtype=np.float32)
y = np.zeros(len(fragments), dtype=np.int64)
for i, fragment in enumerate(fragments):
    for j, char in enumerate(fragment):
        X[i, j, char_to_index[char]] = 1
    y[i] = char_to_index[targets[i]]

# Split into training and test set.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.05, random_state=0)

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))

# Define model.
class LastTimestep(nn.Module):
    def forward(self, inputs):
        # Return hidden state for the last timestep.
        return inputs[1][0][2]

model = nn.Sequential(
    nn.LSTM(encoding_width, 256, num_layers=3, dropout=0.5, batch_first=True),
    LastTimestep(),
    nn.Dropout(0.5), # Add this since PyTorch LSTM does not apply dropout to top layer.
    nn.Linear(256, encoding_width)
)    
model.to(device)

# Loss function and optimizer.
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Train the model.
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset, optimizer, loss_function, 'acc')

# Save the trained model and metadata.
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_index': char_to_index,
    'index_to_char': index_to_char,
    'encoding_width': encoding_width
}, 'model.pth')

