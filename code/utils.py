import numpy as np
from random import shuffle


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def gen(X, Y, batch_size=64):
    indexes = list(range(X.shape[0]))
    while True:
        shuffle(indexes)
        for batch_indexes_1, batch_indexes_2 in zip(
            chunker(indexes, size=batch_size), chunker(indexes[::-1], size=batch_size)
        ):
            alphas = np.random.beta(0.2, 0.2, size=batch_size).tolist()

            X_1 = [X[i, ...] for i in batch_indexes_1]
            X_2 = [X[i, ...] for i in batch_indexes_2]

            Y_1 = [Y[i, ...] for i in batch_indexes_1]
            Y_2 = [Y[i, ...] for i in batch_indexes_2]

            X_batch = [l * a + (1 - l) * b for a, b, l in zip(X_1, X_2, alphas)]

            Y_batch = [l * a + (1 - l) * b for a, b, l in zip(Y_1, Y_2, alphas)]

            yield np.array(X_batch), np.array(Y_batch)
