import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model
import os
import tensorflow as tf
import keras as k
class Emo:

    def _model(self):
        replace = 0.0001
        chroma = Input(shape=(12, 200, 1))
        mfccs = Input(shape=(20, 200, 1))
        rolloff = Input(shape=(200,))
        centroid = Input(shape=(200,))
        # timbre = Input(shape=(10,))
        

        # first branch
        x = Conv2D(filters=4, input_shape=(12, 200, 1), kernel_size=(3, 3), activation='relu')(chroma)
        x = MaxPooling2D((2, 2))(x)
        x = Reshape((1980,))(x)
        x = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=replace))(x)
        # x = Dropout(0.2)(x)

        # second branch
        y = Conv2D(filters=4, input_shape=(20, 200, 1), kernel_size=(3, 3), activation='relu')(mfccs)
        y = MaxPooling2D(pool_size=(2, 2))(y)
        y = Reshape((3564,))(y)
        y = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=replace))(y)

        # third branch
        z = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=replace))(centroid)
        z = Dense(40, activation=tf.keras.layers.LeakyReLU(alpha=replace))(z)
        z = Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=replace))(z)

        # fourth branch
        a = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=replace))(rolloff)
        a = Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=replace))(a)

        # fifth branch
        # b = Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=replace))(timbre)

        combined = Concatenate()([x, y, z, a])  # add c if want

        # final layer
        final = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=replace))(combined)
        final = Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=replace))(final)
        final = Dense(1)(final)

        # opt = tf.keras.optimizers.SGD(learning_rate=0.11, momentum=0.9)
        opt = tf.keras.optimizers.Adadelta(learning_rate=1.5, rho=0.95)

        model = Model(inputs=[chroma, mfccs, rolloff, centroid], outputs=final)  # timbre add tempo here
        model.compile(loss='mean_squared_error', optimizer=opt,
                      metrics=['mse'])
        return model

    def __init__(self):
        self.model = self._model()

    def fit(self, x, y, d_size, b_size = 32, epoch = 30, ckpt = './ckpt', pat = 5):
        steps = d_size//b_size + 1
        steps *= 4
        cp_path = os.path.join(ckpt,"cp.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                         save_weights_only = True,
                                                         verbose = False,
                                                         save_best_only = True,
                                                         monitor="loss",
                                                         save_freq = int(1 * steps))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=pat, verbose = True)

        dataset = tf.data.Dataset.zip((x, y)).shuffle(d_size)

        train_size = int(0.8 * d_size)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(train_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(b_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.repeat()

        val_dataset = val_dataset.batch(b_size)

        self.model.fit(train_dataset, steps_per_epoch=steps, epochs=epoch, validation_data=val_dataset, verbose=True, callbacks=[callback, cp_callback])

    def load(self, cp_path):

        file_path = tf.train.latest_checkpoint(cp_path, latest_filename=None)
        self.model.load_weights(file_path)