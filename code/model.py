from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Convolution1D,
    MaxPool1D,
    GlobalMaxPool1D,
)


def get_model(n_class=5):
    inp = Input(shape=(187, 1))
    x = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="valid")(
        inp
    )
    x = MaxPool1D(pool_size=4)(x)
    x = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(
        x
    )
    x = MaxPool1D(pool_size=4)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation=activations.relu)(x)
    x = Dense(n_class, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
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
