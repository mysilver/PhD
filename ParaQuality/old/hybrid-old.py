import queue
import random

import numpy as np
import tensorflow as tf
from ParaQuality.cnn import create_model as cnn_create_model
from ParaQuality.mlp import get_or_create_features_file as mlp_features
from ParaQuality.mlp import print_statistics
from keras import Input, Model
from keras.layers import Dense, concatenate
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.rnn import get_or_create_features_file as rnn_features, create_model as rnn_create_model


def create_model(manual_features=30, max_length=25, filters=32, embedding_size=300, print_summary=False):
    manual_features = Input(shape=(manual_features,))
    cnn_input_1, cnn_input_2, cnn_output = cnn_create_model(shape=(max_length, embedding_size), filters=filters,
                                                            compile=False)
    rnn_input_1, rnn_input_2, rnn_output = rnn_create_model(max_length=max_length, lstm_size=64, compile=False)

    concatenated = concatenate([cnn_output, rnn_output, manual_features])
    dense_1 = Dense(16, activation="sigmoid")(concatenated)
    predictions = Dense(1, activation="sigmoid")(dense_1)

    model = Model([manual_features, cnn_input_1, cnn_input_2, rnn_input_1, rnn_input_2], predictions)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    batch_size = 128
    epochs = 100
    max_length = 25
    filters = 32
    embedding_size = 300
    callbacks = []
    rnn_feature_path = "../paraphrasing-data/rnn-features.pickle"
    mlp_feature_path = "../paraphrasing-data/features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X1, X2, Y, T = rnn_features(rnn_feature_path, datasets, max_length, False)
    rnn_f = dict(zip(T, zip(X1, X2, Y)))
    mX, mY, mT = mlp_features(mlp_feature_path, datasets, False)
    mlp_f = dict(zip(mT, mX))

    for key in rnn_f:
        rnn_f[key] = rnn_f[key] + (mlp_f[key],)

    T = list(rnn_f.keys())
    X1, X2, Y, X = map(list, zip(*list(rnn_f.values())))

    z = list(zip(X, X1, X2, Y, T))
    random.shuffle(z)
    X, X1, X2, Y, T = zip(*z)

    T = np.array(T)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[2]))

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model(30, max_length, filters=filters, embedding_size=embedding_size)
            model.fit([X[train], X1[train], X2[train], X1[train], X2[train]], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                      verbose=0)
            scores = model.evaluate([X[test], X1[test],  X2[test], X1[test],  X2[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X[test], X1[test],  X2[test], X1[test],  X2[test]])
            y_pred = (y_pred_v >= 0.5)
            tn, fp, fn, tp = print_statistics(Y[test], y_pred)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            y_pred = y_pred.tolist()
            y_test = Y[test]
            t_test = T[test]

            q = queue.PriorityQueue()
            for i in range(len(y_pred_v)):
                pred_class = y_pred[i]
                real_class = y_test[i] == 1

                y = y_pred_v[i][0]
                y_r = y_test[i]

                if pred_class[0] != real_class:
                    q.put((- abs(y - y_r), y_r, y, t_test[i]))

                if q.qsize() == 20:
                    break

            while not q.empty():
                score, real, stimated, text = q.get()
                # print(real, stimated, text)  # features

        print("tn:", ttn, "fp:", tfp, "fn:", tfn, "tp:", ttp)
        print("Accuracy:", (ttp + ttn) / (ttp + ttn + tfp + tfn))
