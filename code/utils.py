import numpy as np


def gen(X, Y, batch_size=64):
    indexes = list(range(X.shape[0]))
    while True:
        batch_indexes_1 = np.random.choice(indexes, size=batch_size).tolist()
        batch_indexes_2 = np.random.choice(indexes, size=batch_size).tolist()
        alphas = np.random.beta(1, 1, size=batch_size).tolist()

        X_1 = [X[i, ...] for i in batch_indexes_1]
        X_2 = [X[i, ...] for i in batch_indexes_2]

        Y_1 = [Y[i, ...] for i in batch_indexes_1]
        Y_2 = [Y[i, ...] for i in batch_indexes_2]

        X_batch = [l * a + (1 - l) * b for a, b, l in zip(X_1, X_2, alphas)]

        Y_batch = [l * a + (1 - l) * b for a, b, l in zip(Y_1, Y_2, alphas)]

        yield np.array(X_batch), np.array(Y_batch)