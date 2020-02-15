import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Convolution1D,
    MaxPool1D,
    GlobalMaxPool1D,
    Lambda,
)


def get_model(n_class=5):
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(inp)
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(n_class, activation=activations.softmax, name="dense_3_mitbih")(
        dense_1
    )

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=["acc"])
    model.summary()
    return model


def get_small_model(n_class=5):
    inp = Input(shape=(187, 1))
    x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(
        inp
    )
    x = MaxPool1D(pool_size=4)(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(
        x
    )
    x = MaxPool1D(pool_size=4)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation=activations.relu)(x)
    x = Dense(n_class, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=["acc"])
    model.summary()
    return model


def custom_loss(y_true, y_pred, mae_weight=0.1):
    return losses.kullback_leibler_divergence(y_true, y_pred) + mae_weight * losses.mae(
        y_true, y_pred
    )


def get_kd_model(n_class=5):
    inp = Input(shape=(187, 1))
    x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(
        inp
    )
    x = MaxPool1D(pool_size=4)(x)
    x = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(
        x
    )
    x = MaxPool1D(pool_size=4)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation=activations.relu)(x)
    x = Dense(n_class, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(
        optimizer=opt, loss=losses.kullback_leibler_divergence, metrics=["acc"]
    )
    model.summary()
    return model
