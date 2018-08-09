import copy

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

seed = 42


def preproc_line(line):
    """
    This is how each entry of the corpus will be preprocessed.
    - Remove whitespaces and newlines
    - Lowercase everything
    - Apostrophes replaced with spaces
    - All other punctuation replaced with spaces
    """
    line = line.strip().lower()

    # Replace apostrophes with empty spaces to handle words like "can't"
    line = line.replace("'", "")

    punctuations = [",", "-", "_", ".", "!", '"',
                   "[", "]", ";", ":", "(", ")",
                    "+", "{", "}", "?", "/", "\\"
                   ]

    # Replace all punctuations with spaces.
    for punc in punctuations:
        line = line.replace(punc, " ")

    # Split line by spaces.
    words = line.split()
    # Special case: Remove empty strings.
    words = [w for w in words if w != ""]

    return words


def preproc(corpus):
    """
    Calls preproc_line on each entry in the corpus
    """
    preprocessed_corpus = [preproc_line(line) for line in corpus]
    return preprocessed_corpus


def create_vocab(preprocessed_corpus):
    """
    Create a vocabulary from the tokenized corpus and mappings:
    word_to_idx & idx_to_word
    """

    # Create a vocab of all the unique words
    vocab = set()

    for sentence in preprocessed_corpus:
    #     Update the vocab with words from sentence
        vocab.update(sentence)

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return (vocab, word_to_idx, idx_to_word)


def skipgram_model(tokenized_corpus, word_to_idx, window_size=2):
    """
    Returns a list of pairs of the form (center, context),
    where the context is within the window size for the given word.

    Each item in the tuple is the index for that word.
    """
    pairs = []

    for sentence in tokenized_corpus:
#         Replace words with their respective indices.
        sentence = [word_to_idx[w] for w in sentence]
        sen_len = len(sentence)

#         Treat each word in sentence as the center.
        for center_pos, center in enumerate(sentence):
#             Slide over the window.
            for slide in range(-window_size, window_size + 1):
                if slide == 0:
                    continue

#                 This is the position of the context word to consider
                context_pos = center_pos + slide

#                 If the context position is out of bounds, ignore
                if context_pos < 0 or context_pos >= sen_len:
                    continue

                context = sentence[context_pos]
                pairs.append((center, context))

#     Return the pairs as an np array
    return np.array(pairs)


def onehot_encode(vsize, word_idx):
    """
    Given a word_idx, this function will return a Tensor of length
    vsize with a 1.0 placed in the corresponding word_idx.
    """
    onehot = torch.zeros(vsize)
    onehot[word_idx] = 1.0

    return onehot


def onehot_encode_word(vsize, word, word_to_idx):
    """
    Given a vsize, word, and word_to_idx, this function will
    compute word_idx and return the onehot_encode vector.
    """
    word_idx = word_to_idx[word]
    return onehot_encode(vsize, word_idx)


# Define our WordEmbeddingNetwork Network Here
class WordEmbeddingNetwork(nn.Module):
    def __init__(self, vocab_size=15, embedding_dim=5):
        super(WordEmbeddingNetwork, self).__init__()
        self.w1 = nn.Linear(vocab_size, embedding_dim)
        self.w2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out1 = self.w1(x)
        out2 = self.w2(out1)

        log_softmax = F.log_softmax(out2, dim=0)
        return log_softmax

    def get_encoder(self):
        """
        Returns a copy of self.w1, the affine map that takes an
        onehot vector as input and outputs the embedding for that
        vector.
        """
        return copy.deepcopy(self.w1)


def train_model(onehot_pairs, vsize, embedding_dim=5, epochs=5,
    learning_rate=1e-3, opt_alg=optim.Adam, criterion=nn.MSELoss()):
    """
    Helper function for training a a model.

    DOES NOT USE BATCHING.
    """

    torch.manual_seed(seed)
    model = WordEmbeddingNetwork(vocab_size=vsize, embedding_dim=embedding_dim)
    optimizer = opt_alg(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = 0.0
        for _, onehot_pair in enumerate(onehot_pairs, 0):
            x_i, y_i = onehot_pair

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_i)
            loss = criterion(outputs, y_i)
            loss.backward()
            optimizer.step()

            loss += loss.item()
        # Print the loss for every 5 epochs
        if (epoch + 1) % 5 == 0:
            print('Epoch: {} loss: {:.3f}'.format(epoch + 1, loss))

    return model


def train_model_batches(trainloader, vsize, embedding_dim=5, epochs=50,
    learning_rate=1e-3, opt_alg=optim.Adam, criterion=nn.MSELoss()):
    """
    Helper function for training a a model.

    USES BATCHING.
    """

    torch.manual_seed(seed)
    model = WordEmbeddingNetwork(vocab_size=vsize, embedding_dim=embedding_dim)
    optimizer = opt_alg(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = 0.0
        for _, batch in enumerate(trainloader):

            X, y = batch

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss += loss.item()

        # Print the loss for every 5 epochs
        if (epoch + 1) % 5 == 0:
            print('Epoch: {} loss: {:.3f}'.format(epoch + 1, loss))

    return model
