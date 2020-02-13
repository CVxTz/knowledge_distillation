import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Lambda


def get_model(do_rate=0.2):
    n_class = 5
    inp = Input(shape=(187, 1))
    x = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    x = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(rate=do_rate)(x)
    x = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=do_rate)(x)

    x = Dense(64, activation=activations.relu, name="dense_1")(x)
    x = Dropout(rate=do_rate)(x)
    x = Dense(64, activation=activations.relu, name="dense_2")(x)
    x = Dropout(rate=do_rate)(x)
    x = Dense(n_class, activation=activations.softmax, name="dense_3_mitbih")(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def get_small_model(do_rate=0.2):
    n_class = 5
    inp = Input(shape=(187, 1))
    x = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    x = MaxPool1D(pool_size=4)(x)
    x = Dropout(rate=do_rate)(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(pool_size=4)(x)
    x = Dropout(rate=do_rate)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=do_rate)(x)
    x = Dense(64, activation=activations.relu)(x)
    x = Dropout(rate=do_rate)(x)
    x = Dense(n_class, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def get_mlp_model(do_rate=0.2):
    n_class = 6
    inp = Input(shape=(561,))
    x = Dense(64, activation=activations.relu)(inp)
    x = Dropout(rate=do_rate)(x)

    x = Dense(32, activation=activations.relu)(x)
    x = Dropout(rate=do_rate)(x)
    x = Dense(n_class, activation=activations.softmax)(x)
    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def get_small_mlp_model(do_rate=0.2):
    n_class = 6
    inp = Input(shape=(561,))
    x = Dense(8, activation=activations.relu)(inp)
    x = Dropout(rate=do_rate)(x)
    x = Dense(n_class, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model