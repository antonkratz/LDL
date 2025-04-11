import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 128
MAX_WORDS = 10000
READ_LINES = 60000
LAYER_SIZE = 256
EMBEDDING_WIDTH = 128
TEST_PERCENT = 0.2
SAMPLE_SIZE = 20
OOV_WORD = 'UNK'
PAD_INDEX = 0
OOV_INDEX = 1
START_INDEX = MAX_WORDS - 2
STOP_INDEX = MAX_WORDS - 1
MAX_LENGTH = 60
SRC_DEST_FILE_NAME = '../data/deu.txt'

# A simple tokenizer and helper functions.
def text_to_word_sequence(text): # {{{
    # Convert text to lowercase and split on word characters.
    return re.findall(r'\w+', text.lower())

class SimpleTokenizer:
    def __init__(self, num_words, oov_token):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        # texts is a list of lists of tokens.
        for tokens in texts:
            for word in tokens:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        # Sort words by frequency (high to low).
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Reserve indices: 0 for PAD and 1 for OOV.
        limit = self.num_words
        for i, (word, count) in enumerate(sorted_words):
            if i >= limit:
                break
            self.word_index[word] = i + 2  # starting index 2
        self.index_word = {index: word for word, index in self.word_index.items()}

    def texts_to_sequences(self, tokens):
        # tokens is a list of tokens.
        seq = []
        for word in tokens:
            seq.append(self.word_index.get(word, OOV_INDEX))
        return seq
# }}}

def tokenize(sequences): # {{{
    tokenizer = SimpleTokenizer(num_words=MAX_WORDS - 2, oov_token=OOV_WORD)
    tokenizer.fit_on_texts(sequences)
    token_sequences = [tokenizer.texts_to_sequences(seq) for seq in sequences]
    return tokenizer, token_sequences
# }}}

def pad_sequences(sequences, padding='pre', maxlen=None): # {{{
    if maxlen is None:
        maxlen = max(len(s) for s in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'post':
                seq = seq + [PAD_INDEX] * (maxlen - len(seq))
            else:
                seq = [PAD_INDEX] * (maxlen - len(seq)) + seq
        else:
            seq = seq[:maxlen]
        padded.append(seq)
    return np.array(padded)
# }}}

# Function to read file.
def read_file_combined(file_name, max_len): # {{{
    src_word_sequences = []
    dest_word_sequences = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == READ_LINES:
                break
            pair = line.strip().split('\t')
            if len(pair) < 2:
                continue
            # For source, take from the second column; for destination, from the first.
            src_tokens = text_to_word_sequence(pair[1])[:max_len]
            dest_tokens = text_to_word_sequence(pair[0])[:max_len]
            src_word_sequences.append(src_tokens)
            dest_word_sequences.append(dest_tokens)
    return src_word_sequences, dest_word_sequences
# }}}

# Functions to convert tokens back to words.
def tokens_to_words(tokenizer, seq): # {{{
    word_seq = []
    for index in seq:
        if index == PAD_INDEX:
            word_seq.append('PAD')
        elif index == OOV_INDEX:
            word_seq.append(OOV_WORD)
        elif index == START_INDEX:
            word_seq.append('START')
        elif index == STOP_INDEX:
            word_seq.append('STOP')
        else:
            word_seq.append(tokenizer.index_word.get(index, OOV_WORD))
    print(word_seq)
# }}}

# Read file and tokenize.
src_seq, dest_seq = read_file_combined(SRC_DEST_FILE_NAME, MAX_LENGTH)
src_tokenizer, src_token_seq = tokenize(src_seq)
dest_tokenizer, dest_token_seq = tokenize(dest_seq)

# Prepare training data.
dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
dest_input_token_seq = [[START_INDEX] + x for x in dest_target_token_seq]
src_input_data = pad_sequences(src_token_seq, padding='pre')
dest_input_data = pad_sequences(dest_input_token_seq, padding='post')
dest_target_data = pad_sequences(dest_target_token_seq, padding='post', maxlen=len(dest_input_data[0]))

# Convert to same precision as model.
src_input_data = src_input_data.astype(np.int64)
dest_input_data = dest_input_data.astype(np.int64)
dest_target_data = dest_target_data.astype(np.int64)

# Split into training and test set.
rows = src_input_data.shape[0]
all_indices = list(range(rows))
test_rows = int(rows * TEST_PERCENT)
test_indices = random.sample(all_indices, test_rows)
train_indices = [x for x in all_indices if x not in test_indices]

train_src_input_data = src_input_data[train_indices]
train_dest_input_data = dest_input_data[train_indices]
train_dest_target_data = dest_target_data[train_indices]

test_src_input_data = src_input_data[test_indices]
test_dest_input_data = dest_input_data[test_indices]
test_dest_target_data = dest_target_data[test_indices]

# Create a sample of the test set that we will inspect in detail.
sample_indices = random.sample(list(range(test_rows)), SAMPLE_SIZE)
sample_input_data = test_src_input_data[sample_indices]
sample_target_data = test_dest_target_data[sample_indices]

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_src_input_data),
                         torch.from_numpy(train_dest_input_data),
                         torch.from_numpy(train_dest_target_data))
testset = TensorDataset(torch.from_numpy(test_src_input_data),
                        torch.from_numpy(test_dest_input_data),
                        torch.from_numpy(test_dest_target_data))

# Define models.
class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05)
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, LAYER_SIZE, num_layers=2, batch_first=True)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        _, state = self.lstm_layers(x)
        return state  # Returning (hidden, cell)

class DecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None
        self.use_state = False
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05)
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, LAYER_SIZE, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(LAYER_SIZE, MAX_WORDS)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        if self.use_state and self.state is not None:
            x, new_state = self.lstm_layers(x, self.state)
        else:
            x, new_state = self.lstm_layers(x)
        self.state = (new_state[0].detach().clone(), new_state[1].detach().clone())
        x = self.output_layer(x)
        return x

    def set_state(self, state):
        self.state = state
        self.use_state = True

    def get_state(self):
        return self.state

    def clear_state(self):
        self.use_state = False

encoder_model = EncoderModel()
decoder_model = DecoderModel()

# Loss functions and optimizer.
encoder_optimizer = torch.optim.RMSprop(encoder_model.parameters(), lr=0.001)
decoder_optimizer = torch.optim.RMSprop(decoder_model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Transfer model to GPU.
encoder_model.to(device)
decoder_model.to(device)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

# Train and test repeatedly.
for i in range(EPOCHS):
    encoder_model.train()
    decoder_model.train()
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    train_elems = 0
    for src_inputs, dest_inputs, dest_targets in trainloader:
        src_inputs, dest_inputs, dest_targets = src_inputs.to(device), dest_inputs.to(device), dest_targets.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_state = encoder_model(src_inputs)
        decoder_model.set_state(encoder_state)
        outputs = decoder_model(dest_inputs)
        loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
        _, indices = torch.max(outputs.data, 2)
        train_correct += (indices == dest_targets).sum().item()
        train_elems += indices.numel()
        train_batches += 1
        train_loss += loss.item()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    train_loss = train_loss / train_batches
    train_acc = train_correct / train_elems

    # Evaluate on test dataset.
    encoder_model.eval()
    decoder_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_batches = 0
    test_elems = 0
    with torch.no_grad():
        for src_inputs, dest_inputs, dest_targets in testloader:
            src_inputs, dest_inputs, dest_targets = src_inputs.to(device), dest_inputs.to(device), dest_targets.to(device)
            encoder_state = encoder_model(src_inputs)
            decoder_model.set_state(encoder_state)
            outputs = decoder_model(dest_inputs)
            loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
            _, indices = torch.max(outputs, 2)
            test_correct += (indices == dest_targets).sum().item()
            test_elems += indices.numel()
            test_batches += 1
            test_loss += loss.item()

    test_loss = test_loss / test_batches
    test_acc = test_correct / test_elems
    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')

    # Loop through samples to inspect results.
    for test_input, test_target in zip(sample_input_data, sample_target_data):
        # Run a single sentence through the encoder.
        x = np.reshape(test_input, (1, -1))
        inputs = torch.from_numpy(x).to(device)
        last_states = encoder_model(inputs)

        decoder_model.set_state(last_states)
        prev_word_index = START_INDEX
        pred_seq = []
        for j in range(MAX_LENGTH):
            x = np.reshape(np.array(prev_word_index), (1, 1))
            inputs = torch.from_numpy(x).to(device)
            outputs = decoder_model(inputs)
            preds = outputs.cpu().detach().numpy()[0][0]
            prev_word_index = preds.argmax()
            pred_seq.append(prev_word_index)
            if prev_word_index == STOP_INDEX:
                break
        tokens_to_words(src_tokenizer, test_input)
        tokens_to_words(dest_tokenizer, test_target)
        tokens_to_words(dest_tokenizer, pred_seq)
        print('\n\n')

