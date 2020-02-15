import numpy as np
from random import shuffle
from collections import Counter


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def gen(X, Y, batch_size=64):
    indexes = list(range(X.shape[0]))
    while True:
        shuffle(indexes)
        alphas = np.random.beta(0.2, 0.2, size=X.shape[0]).tolist()
        for batch_indexes_1, batch_indexes_2, alphas_batch in zip(
            chunker(indexes, size=batch_size),
            chunker(indexes[::-1], size=batch_size),
            chunker(alphas, size=batch_size),
        ):

            X_1 = [X[i, ...] for i in batch_indexes_1]
            X_2 = [X[i, ...] for i in batch_indexes_2]

            Y_1 = [Y[i, ...] for i in batch_indexes_1]
            Y_2 = [Y[i, ...] for i in batch_indexes_2]

            X_batch = [l * a + (1 - l) * b for a, b, l in zip(X_1, X_2, alphas_batch)]

            Y_batch = [l * a + (1 - l) * b for a, b, l in zip(Y_1, Y_2, alphas_batch)]

            yield np.array(X_batch), np.array(Y_batch)


def get_mixup(X, Y):
    indexes = list(range(X.shape[0]))
    Y_l = np.argmax(Y, axis=-1).tolist()
    cnt = Counter(Y_l)
    Y_w = np.array([1./cnt[a] for a in Y_l])
    Y_w = Y_w/np.sum(Y_w)
    sampled_indexes = np.random.choice(indexes, size=X.shape[0], p=Y_w, replace=True)
    alphas = np.random.beta(0.1, 0.1, size=X.shape[0]).tolist()
    X_1 = [X[i, ...] for i in sampled_indexes]
    X_2 = [X[i, ...] for i in sampled_indexes[::-1]]

    Y_1 = [Y[i, ...] for i in sampled_indexes]
    Y_2 = [Y[i, ...] for i in sampled_indexes[::-1]]

    X_new = [l * a + (1 - l) * b for a, b, l in zip(X_1, X_2, alphas)]

    Y_new = [l * a + (1 - l) * b for a, b, l in zip(Y_1, Y_2, alphas)]

    return X_new, Y_new
