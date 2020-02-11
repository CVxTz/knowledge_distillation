import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Lambda


def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def get_model(do_rate=0.2, use_normal_dropout=True):
    nclass = 5
    inp = Input(shape=(187, 1))
    x = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    x = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)
    x = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)

    x = Dense(64, activation=activations.relu, name="dense_1")(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)
    x = Dense(64, activation=activations.relu, name="dense_2")(x)
    x = Dropout(rate=do_rate)(x) if use_normal_dropout else PermaDropout(rate=do_rate)(x)
    x = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
