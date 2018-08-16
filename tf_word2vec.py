import copy

import numpy as np

import tensorflow as tf
from keras.models import Sequential
import keras.layers as layers
from keras.layers.embeddings import Embedding

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
    onehot = np.zeros(vsize)
    onehot[word_idx] = 1.0

    return onehot


def onehot_encode_word(vsize, word, word_to_idx):
    """
    Given a vsize, word, and word_to_idx, this function will
    compute word_idx and return the onehot_encode vector.
    """
    word_idx = word_to_idx[word]
    return onehot_encode(vsize, word_idx)



def train_model(onehot_pairs, vsize, embedding_dim=5, epochs=5,
    learning_rate=1e-3, opt_alg="rmsprop", loss_func="mse"):
    """
    Helper function for training a a model.

    DOES NOT USE BATCHING.
    """

    # TODO: Fix this model, it isn't working properly at the moment
    # Need to use keras.layers.embeddings import Embedding
    model = Sequential([
        Embedding(vsize, embedding_dim)
    ])

    model.compile(optimizer=opt_alg,
                  loss=loss_func,
                  metrics=["accuracy"])

    X = list(map(lambda pair: pair[0], onehot_pairs))
    y = list(map(lambda pair: pair[1], onehot_pairs))

    print(type(X))
    print(type(X[0]))
    print(X[0].shape)

    print(type(y))
    print(type(y[0]))
    print(y[0].shape)

    model.fit(X, y, epochs=epochs)

    return model
