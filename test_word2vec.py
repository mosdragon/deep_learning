from word2vec import *

import torch.nn as nn

# Use cosine similarity for evaluation.
cos = nn.CosineSimilarity()

# Example corpus
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
    'madrid is spain capital',
]

# Preprocess the corpus.
preprocessed_corpus = preproc(corpus)
(vocab, word_to_idx, idx_to_word) = create_vocab(preprocessed_corpus)
vsize = len(vocab)

# Prepare our training data
pairs = skipgram_model(preprocessed_corpus, word_to_idx, window_size=2)
onehot_pairs = [(onehot_encode(vsize, x), onehot_encode(vsize, y)) for x, y in pairs]

# Batches up our pairs of data
batch_size = 3
trainloader = DataLoader(onehot_pairs, batch_size=batch_size, shuffle=True)

# Training
model = train_model_batches(trainloader, vsize, embedding_dim=5, epochs=50,
    learning_rate=1e-3, opt_alg=optim.Adam,
    criterion=nn.MultiLabelSoftMarginLoss())

# Evaluation function.
def argmax_cos(estimate):
    cos = nn.CosineSimilarity()
    costs = [(word, cos(estimate, enc)) for word, enc in encodings.items()]
    item = max(costs, key=lambda x: x[1])
    return item


# Extract the encoder from the model.
encoder = model.get_encoder()

# Create encodings of the words to their respective vectors
encodings = {w: encoder(onehot_encode(vsize, idx)) for w, idx in word_to_idx.items()}
# Reshape those encodings to be of dimension (embedding_dim, -1)
encodings = {w: e.reshape(1, -1) for w, e in encodings.items()}


print("=========================")
queen = encodings["queen"]
print("Similarity <queen> to <king - man + woman>: ",
    cos(queen, encodings["king"] - encodings["man"] + encodings["woman"]))

print(argmax_cos(encodings["king"] - encodings["man"] + encodings["woman"]))

print("=========================")
paris = encodings["paris"]
print("Similarity <paris> to <berlin - germany + france>: ",
    cos(paris, encodings["berlin"] - encodings["germany"] + encodings["france"]))
print(argmax_cos(encodings["berlin"] - encodings["germany"] + encodings["france"]))
