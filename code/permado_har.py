import json
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import get_mlp_model

if __name__ == "__main__":
    file_path = "permado_har.h5"

    X_train = np.loadtxt("../input/har/train/X_train.txt", dtype=np.float)
    Y_train = np.loadtxt("../input/har/train/y_train.txt", dtype=np.float)

    print(X_train.shape)

    X_eval = np.loadtxt("../input/har/test/X_test.txt", dtype=np.float)
    Y_eval = np.loadtxt("../input/har/test/y_test.txt", dtype=np.float)

    X_test, X_val, Y_test, Y_val = train_test_split(X_eval, Y_eval, test_size=0.2, random_state=1337)

    Y_train = np.array(Y_train).astype(np.int8)[..., np.newaxis] - 1
    X_train = np.array(X_train)

    Y_val = np.array(Y_val).astype(np.int8)[..., np.newaxis] - 1
    X_val = np.array(X_val)

    Y_test = np.array(Y_test).astype(np.int8)[..., np.newaxis] - 1
    X_test = np.array(X_test)

    print(min(Y_train), max(Y_train), X_train.shape, Y_train.shape)

    model = get_mlp_model(use_normal_dropout=False)

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode="min")
    reduce = ReduceLROnPlateau(monitor="val_loss", patience=10, min_lr=1e-7, mode="min")
    early = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1000, verbose=2, batch_size=64,
              callbacks=[checkpoint, reduce, early])

    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    acc = accuracy_score(Y_test, pred_test)

    print("acc :", acc)
    print("f1 :", f1)

    repeat = 100
    pred_test = 0

    for _ in range(repeat):
        pred_test += model.predict(X_test)

    pred_test = np.argmax(pred_test, axis=-1)

    ensemble_f1 = f1_score(Y_test, pred_test, average="macro")

    ensemble_acc = accuracy_score(Y_test, pred_test)

    print("ensemble acc :", ensemble_acc)
    print("ensemble f1 :", ensemble_f1)

    rnd = np.random.randint(1, 100000)
    os.makedirs('../output/har/', exist_ok=True)

    with open('../output/har/permado_performance_%s.json' % int(rnd), 'w') as f:
        json.dump({"acc": acc, "f1": f1, "ensemble_acc": ensemble_acc, "ensemble_f1": ensemble_f1}, f, indent=4)
