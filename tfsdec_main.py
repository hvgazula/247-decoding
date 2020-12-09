'''
Filename: /scratch/gpfs/hgazula/brain2text/tfsdec_main.py
Path: /scratch/gpfs/hgazula/brain2text
Created Date: Sunday, December 6th 2020, 8:59:42 am
Author: Harshvardhan Gazula

Copyright (c) 2020 Your Company
'''
import argparse
import json
import os
import pickle
import random as python_random
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, GlobalMaxPooling1D, Input,
                                     LayerNormalization, LocallyConnected1D,
                                     MaxPooling1D, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from transformers import TFBertForMaskedLM

from evaluate import evaluate_roc, evaluate_topk


def set_seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(123)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(1234)


def arg_parser():
    args = argparse.Namespace()
    args.patience = 150
    args.verbose = 1
    args.conv_filters = 128
    args.reg = 0.35
    args.dropout = 0.2
    args.reg_head = 0
    args.lr = 0.00025
    args.fine_epochs = 1000
    args.lm_head = 0
    args.seed = 0
    args.batch_size = 1024
    args.model = '247_decoding_test'
    args.lag = -160
    args.half_window = 512

    # lags = np.arange(-1000, 1001, 100).tolist()
    # lags = np.arange(-800, 8001, 32).tolist()

    return args


def load_pickles():
    with open('625_binned_signal.pkl', 'rb') as fh:
        signal_d = pickle.load(fh)

    with open('625_both_labels_MWF30.pkl', 'rb') as fh:
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
    stitch_index.insert(0, 0)

    # The first 64 electrodes correspond to the hemisphere of interest
    signals = signals[:, :64]
    # print(signals.shape)

    # The labels have been stemmed using Porter Stemming Algorithm

    return signals, stitch_index, label_folds


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
    """Averages model weights across training trajectory, starting at designated
    epoch."""


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


# Define language model decoder
def language_decoder(args):
    lang_model = TFBertForMaskedLM.from_pretrained(
        args.model_name, cache_dir='/scratch/gpfs/zzada/cache-tf')
    d_size = lang_model.config.hidden_size
    v_size = lang_model.config.vocab_size

    lang_decoder = lang_model.mlm
    lang_decoder.trainable = False

    inputs = Input((d_size, ))
    x = Reshape((1, d_size))(inputs)
    x = lang_decoder(x)
    x = Reshape((v_size, ))(x)
    # x = Lambda(lambda z: tf.gather(z, vocab_indices, axis=-1))(x)
    x = Activation('softmax')(x)
    lm_decoder = Model(inputs=inputs, outputs=x)
    lm_decoder.summary()
    return lm_decoder


def get_decoder():
    if args.lm_head:
        return language_decoder()
    else:
        return Dense(n_classes,
                     kernel_regularizer=l2(args.reg_head),
                     activation='softmax')


def extract_signal_from_fold(examples, stitch_index, args):

    lag_in_bin_dim = args.lag // 32
    half_window = args.half_window // 32

    x, w = [], []
    for label in examples:
        bin_index = label['onset'] // 32
        bin_rank = (np.array(stitch_index) < bin_index).nonzero()[0][-1]
        bin_start = stitch_index[bin_rank]
        bin_stop = stitch_index[bin_rank + 1]

        left_edge = bin_index + lag_in_bin_dim - half_window
        right_edge = bin_index + lag_in_bin_dim + half_window

        if (left_edge < bin_start) or (right_edge > bin_stop):
            continue
        else:
            x.append(signals[left_edge:right_edge, :])
            w.append(label['word'])

    x = np.stack(x, axis=0)
    w = np.array(w)

    return x, w


if __name__ == '__main__':

    set_seed()
    args = arg_parser()

    # Logistical things
    save_dir = os.path.join(os.getcwd(), args.model) + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    signals, stitch_index, label_folds = load_pickles()

    train_fold = [
        example for example in label_folds if example['fold0'] == 'train'
    ]
    dev_fold = [
        example for example in label_folds if example['fold0'] == 'dev'
    ]
    test_fold = [
        example for example in label_folds if example['fold0'] == 'test'
    ]

    # Decoding starts here
    x_train, w_train = extract_signal_from_fold(train_fold, stitch_index, args)
    x_dev, w_dev = extract_signal_from_fold(dev_fold, stitch_index, args)
    x_test, w_test = extract_signal_from_fold(test_fold, stitch_index, args)

    # Determine indexing
    word2index = {w: i for i, w in enumerate(sorted(set(w_train.tolist())))}
    index2word = {i: word for word, i in word2index.items()}

    y_train = np.array([word2index[w] for w in w_train])
    y_dev = np.array([word2index[w] for w in w_dev])
    y_test = np.array([word2index[w] for w in w_test])

    n_classes = np.unique(y_train).size

    print('X train, dev, test:', x_train.shape, x_dev.shape, x_test.shape)
    print('Y train, dev, test:', y_train.shape, y_dev.shape, y_test.shape)
    print('W train, dev, test:', w_train.shape, w_dev.shape, w_test.shape)
    # print('Z train, dev, test:', z_train.shape, z_dev.shape, z_test.shape)
    print('n_classes:', n_classes,
          np.unique(y_dev).size,
          np.unique(y_test).size)

    model = pitom([x_train.shape[1:]], n_classes=None)
    optimizer = Adam(lr=args.lr)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.CosineSimilarity()])

    # model.summary()

    # -------------------------------------------------------------------------
    # >> Classification training
    # -------------------------------------------------------------------------
    corrs = {}
    test_result = {}
    main_history = {}

    histories = []
    test_results = []
    fold_results = []
    i = 0
    # Add the decoder, LM head or just a new layer
    if args.fine_epochs > 0:
        print('I am inside fine_epochs')
        model2 = Model(inputs=model.input, outputs=get_decoder()(model.output))
        model2.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='top1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision')
            ])

        with open(os.path.join(save_dir, 'model2-summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model2.summary()

        stopper = EarlyStopping(monitor='val_top1',
                                mode='max',
                                patience=args.patience,
                                restore_best_weights=True,
                                verbose=args.verbose)

        history = model2.fit(
            x=x_train,
            y=to_categorical(y_train, n_classes),
            epochs=args.fine_epochs,
            batch_size=args.batch_size,
            validation_data=[x_dev, to_categorical(y_dev, n_classes)],
            callbacks=[stopper],
            verbose=args.verbose)

        main_history.update(history.history)

    histories.append(main_history)

    res, res2 = {}, {}
    if args.lm_head or args.fine_epochs > 0:
        # Evaluate end to end on test set
        testset = model2.evaluate(x_test,
                                  to_categorical(y_test, n_classes),
                                  verbose=args.verbose)
        test_result2 = {
            metric: float(result)
            for metric, result in zip(model2.metrics_names, testset)
        }
        test_result.update(test_result2)

        test_results.append(test_result)

        # Prune predictions to only ones that are in w_test, create a mapping
        # of i2w and recreate y_test.
        predictions = model2.predict(x_test)
        if args.lm_head:
            test_vocab = sorted(set(w_test))
            test_indices = [word2index[w] for w in test_vocab]
            test_w2i = {w: i for i, w in enumerate(test_vocab)}
            test_i2w = {i: w for w, i in test_w2i.items()}

            predictions = predictions[:, test_indices]
            y_test_pruned = [test_w2i[w] for w in w_test]
            y_train_freq = Counter(y_test_pruned)
        else:
            y_train_freq = Counter(y_train)
            y_test_pruned = y_test
            test_i2w = index2word

        res = evaluate_topk(predictions,
                            to_categorical(y_test_pruned,
                                           predictions.shape[1]),
                            test_i2w,
                            y_train_freq,
                            save_dir,
                            min_train=0,
                            prefix='eval_',
                            suffix=f'-fold_{i}-test',
                            title=args.model)

        res2 = evaluate_roc(predictions,
                            to_categorical(y_test_pruned,
                                           predictions.shape[1]),
                            test_i2w,
                            y_train_freq,
                            save_dir,
                            min_train=0,
                            suffix=f'-fold_{i}-test',
                            title=args.model)

        # Just to check
        guesses = predictions.argmax(axis=-1)
        accuracy = (guesses == y_test_pruned).sum() / len(y_test_pruned)
        res['manual_top1'] = accuracy

    # Store final value of each dev metric, then test metrics
    values = {k: float(v[-1]) for k, v in main_history.items()}
    values.update({f'test_{metric}': v for metric, v in test_result.items()})
    values.update(corrs)  # add our metrics

    values['n_classes'] = np.unique(y_train).size
    values['n_classes_dev'] = np.unique(y_dev).size
    values['n_classes_test'] = np.unique(y_test).size
    values['n_train'] = x_train.shape[0]
    values['n_dev'] = x_dev.shape[0]
    values['n_test'] = x_test.shape[0]
    values.update(res)
    values.update(res2)

    fold_results.append(values)
    print(json.dumps(values, indent=2))
