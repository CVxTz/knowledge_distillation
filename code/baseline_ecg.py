import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import get_model

if __name__ == "__main__":
    file_path = "baseline.h5"

    df_train = pd.read_csv("../input/mitbih_train.csv", header=None)
    df_test = pd.read_csv("../input/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1337)

    model = get_model()

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
