'''
Filename: /scratch/gpfs/hgazula/brain2text/tfsdec_main.py
Path: /scratch/gpfs/hgazula/brain2text
Created Date: Sunday, December 6th 2020, 8:59:42 am
Author: Harshvardhan Gazula

Copyright (c) 2020 Your Company
'''
import pickle
import sys

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, GlobalMaxPooling1D, Input,
                                     LayerNormalization, LocallyConnected1D,
                                     MaxPooling1D, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


args = {}



with open('625_binned_signal.pkl', 'rb') as fh:
    signal_d = pickle.load(fh)

with open('625_label_folds.pkl', 'rb') as fh:
    label_folds = pickle.load(fh)

print('Signals pickle info')
for key in signal_d.keys():
    print(
        f'key: {key}, \t type: {type(signal_d[key])}, \t shape: {len(signal_d[key])}'
    )

assert signal_d['binned_signal'].shape[0] == signal_d['bin_stitch_index'][
    -1], 'Error: Incorrect Stitching'
assert signal_d['binned_signal'].shape[1] == len(
    signal_d['electrodes']), 'Error: Incorrect number of electrodes'

signals = signal_d['binned_signal']
stitch_index = signal_d['bin_stitch_index']
# print(signals.shape)

# The first 64 electrodes correspond to the hemisphere of interest
signals = signals[:, :64]
# print(signals.shape)

# The labels have been stemmed using Porter Stemming Algorithm
"""Decoding starts here"""
# lags = np.arange(-1000, 1001, 100).tolist()
# lags = np.arange(-800, 8001, 32).tolist()

lag = -160
lag_in_bin_dim = lag // 32
half_window = 512 // 32

stitch_index.insert(0, 0)

test_X, test_y = [], []
for label in label_folds['fold4']:
    bin_index = label['onset'] // 32
    bin_rank = (np.array(stitch_index) < bin_index).nonzero()[0][-1]
    bin_start = stitch_index[bin_rank]
    bin_stop = stitch_index[bin_rank + 1]

    left_edge = bin_index + lag_in_bin_dim - half_window
    right_edge = bin_index + lag_in_bin_dim + half_window

    if (left_edge < bin_start) or (right_edge > bin_stop):
        continue
    else:
        test_X.append(signals[left_edge:right_edge, :])
        test_y.append(label['word'])

val_X, val_y = [], []
for label in label_folds['fold3']:
    bin_index = label['onset'] // 32
    bin_rank = (np.array(stitch_index) < bin_index).nonzero()[0][-1]
    bin_start = stitch_index[bin_rank]
    bin_stop = stitch_index[bin_rank + 1]

    left_edge = bin_index + lag_in_bin_dim - half_window
    right_edge = bin_index + lag_in_bin_dim + half_window

    if (left_edge < bin_start) or (right_edge > bin_stop):
        continue
    else:
        val_X.append(signals[left_edge:right_edge, :])
        val_y.append(label['word'])

train_X, train_y = [], []

for fold in [label_folds['fold0'], label_folds['fold1'], label_folds['fold2']]:
    for label in fold:
        bin_index = label['onset'] // 32
        bin_rank = (np.array(stitch_index) < bin_index).nonzero()[0][-1]
        bin_start = stitch_index[bin_rank]
        bin_stop = stitch_index[bin_rank + 1]

        left_edge = bin_index + lag_in_bin_dim - half_window
        right_edge = bin_index + lag_in_bin_dim + half_window

        if (left_edge < bin_start) or (right_edge > bin_stop):
            continue
        else:
            train_X.append(signals[left_edge:right_edge, :])
            train_y.append(label['word'])

print(len(train_X), len(train_y))
print(len(val_X), len(val_y))
print(len(test_X), len(test_y))


def pitom(input_shapes, n_classes):
    '''
    pitom1: [(128,9), (128,9), ('max', 2), (128, 4)]; DP 0.1 LR 2e-5
    pitom2: [(128,9), ('max',2), (128,4)]; DP 0.1, REG .05; LR 1e-3
    input_shapes = (input_shape_cnn, input_shape_emb)
    '''

    desc = [(args.conv_filters, 3), ('max', 2), (args.conv_filters, 2)]

    input_cnn = Input(shape=input_shapes[0])

    prev_layer = input_cnn
    for filters, kernel_size in desc:
        if filters == 'max':
            prev_layer = MaxPooling1D(pool_size=kernel_size,
                                      strides=None,
                                      padding='same')(prev_layer)
        else:
            # Add a convolution block
            prev_layer = Conv1D(filters,
                                kernel_size,
                                strides=1,
                                padding='valid',
                                use_bias=False,
                                kernel_regularizer=l2(args.reg),
                                kernel_initializer='glorot_normal')(prev_layer)
            prev_layer = Activation('relu')(prev_layer)
            prev_layer = BatchNormalization()(prev_layer)
            prev_layer = Dropout(args.dropout)(prev_layer)

    # Add final conv block
    prev_layer = LocallyConnected1D(
        filters=args.conv_filters,
        kernel_size=2,
        strides=1,
        padding='valid',
        kernel_regularizer=l2(args.reg),
        kernel_initializer='glorot_normal')(prev_layer)
    prev_layer = BatchNormalization()(prev_layer)
    prev_layer = Activation('relu')(prev_layer)

    cnn_features = GlobalMaxPooling1D()(prev_layer)

    output = cnn_features
    if n_classes is not None:
        output = LayerNormalization()(Dense(units=n_classes,
                                            kernel_regularizer=l2(
                                                args.reg_head),
                                            activation='tanh')(cnn_features))

    model = Model(inputs=input_cnn, outputs=output)
    return model


class WeightAverager(tf.keras.callbacks.Callback):
    """Averages model weights across training trajectory, starting at designated epoch."""


def __init__(self, epoch_count, patience):
    super(WeightAverager, self).__init__()
    self.epoch_count = min(epoch_count, 2 * patience)
    self.weights = []
    self.patience = patience


def on_train_begin(self, logs=None):
    print('Weight averager over last {} epochs.'.format(self.epoch_count))


def on_epoch_end(self, epoch, logs=None):
    if len(self.weights) and len(
            self.weights) == self.patience + self.epoch_count / 2:
        self.weights.pop(0)
    self.weights.append(self.model.get_weights())


def on_train_end(self, logs=None):
    if self.weights:
        self.best_weights = np.asarray(self.model.get_weights())
        w = 0
        p = 0
        for p, nw in enumerate(self.weights):
            w = (w * p + np.asarray(nw)) / (p + 1)
            if p >= self.epoch_count:
                break
        self.model.set_weights(w)
        print('Averaged {} weights.'.format(p + 1))
