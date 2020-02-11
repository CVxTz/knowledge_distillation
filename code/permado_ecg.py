import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import get_model

if __name__ == "__main__":
    file_path = "permado.h5"

    df_train = pd.read_csv("../input/mitbih_train.csv", header=None)

    df_train = df_train.sample(n=10000, random_state=1337)

    df = pd.read_csv("../input/mitbih_test.csv", header=None)

    df_test, df_val = train_test_split(df, test_size=0.2, random_state=1337)

    Y_train = np.array(df_train[187].values).astype(np.int8)
    X_train = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_val= np.array(df_val[187].values).astype(np.int8)
    X_val = np.array(df_val[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    model = get_model(use_normal_dropout=False)

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode="min")
    reduce = ReduceLROnPlateau(monitor="val_loss", patience=10, min_lr=1e-7, mode="min")
    early = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1000, verbose=1, batch_size=64,
              callbacks=[checkpoint, reduce, early])

    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    acc = accuracy_score(Y_test, pred_test)

    print("acc :", acc)
    print("f1 :", f1)

    repeat = 10
    pred_test = 0

    for _ in range(repeat):
        pred_test += model.predict(X_test)

    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    acc = accuracy_score(Y_test, pred_test)

    print("ensemble acc :", acc)
    print("ensemble f1 :", f1)
