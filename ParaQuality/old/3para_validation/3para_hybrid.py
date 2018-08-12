import os
import pickle
import random

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPool1D, Flatten, dot, Dropout, Bidirectional, LSTM
from keras.utils import to_categorical
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.rnn import sentence_to_vec
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks


def to_vector(dataset, max_vec, save=None):
    X1, X2, X3, X4, Y, T = [], [], [], [], [], []

    # vectorize
    for i, expression in enumerate(dataset):
        expression_vec = sentence_to_vec(expression, max_vec)
        for instance in dataset[expression]:
            paraphrase_1 = instance[0]
            paraphrase_2 = instance[1]
            paraphrase_3 = instance[2]
            X1.append(expression_vec)
            X2.append(sentence_to_vec(paraphrase_1[0], max_vec))
            X3.append(sentence_to_vec(paraphrase_2[0], max_vec))
            X4.append(sentence_to_vec(paraphrase_3[0], max_vec))
            code = "0" if paraphrase_1[1] == 'invalid' else "1"
            code += "0" if paraphrase_2[1] == 'invalid' else "1"
            code += "0" if paraphrase_3[1] == 'invalid' else "1"
            y = [0, 0, 0, 0, 0, 0, 0, 0]
            # y[int(code, 2)] = 1
            Y.append(int(code, 2))
            T.append(expression + "<==>" + paraphrase_1[0] + "<==>" + paraphrase_2[0] + "<==>" + paraphrase_3[0])
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((X1, X2, X3, X4, Y, T), open(save, 'wb'))

    return np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets, max_vec, shuffle=True):
    # Read the file in the following format:
    # expression    paraphrase  [valid/invalid]
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X1, X2, X3, X4, Y, T = pickle.load(f)
    else:
        dataset = read_paraphrased_tsv_files(datasets, processor=remove_marks, by_user=True)
        X1, X2, X3, X4, Y, T = to_vector(dataset, max_vec, save=dataset_feature_path)

    if shuffle:
        z = list(zip(X1, X2, X3, X4, Y, T))
        random.shuffle(z)
        X1, X2, X3, X4, Y, T = zip(*z)

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    Y = np.array(Y)
    T = np.array(T)

    print("Input : 4 * ", X1.shape)
    print("Output :", Y.shape)
    return X1, X2, X3, X4, Y, T


def create_model(manual_features=30, max_length=25, filters=32, embedding_size=300, lstm_size=64, print_summary=False):
    # manual_features_p1 = Input(shape=(manual_features,))
    # manual_features_p2 = Input(shape=(manual_features,))
    # manual_features_p3 = Input(shape=(manual_features,))
    input_1 = Input(shape=(max_length, embedding_size))

    input_p1 = Input(shape=(max_length, embedding_size))
    input_p2 = Input(shape=(max_length, embedding_size))
    input_p3 = Input(shape=(max_length, embedding_size))

    conv = Conv1D(filters=filters, kernel_size=4)
    conv_out_1 = conv(input_1)
    conv_out_p1 = conv(input_p1)
    conv_out_p2 = conv(input_p2)
    conv_out_p3 = conv(input_p3)

    max_layer = MaxPool1D(pool_size=4)
    maxp_1 = max_layer(conv_out_1)
    maxp_p1 = max_layer(conv_out_p1)
    maxp_p2 = max_layer(conv_out_p2)
    maxp_p3 = max_layer(conv_out_p3)

    flatter = Flatten()
    flat_1, flat_p1, flat_p2, flat_p3 = flatter(maxp_1), flatter(maxp_p1), flatter(maxp_p2), flatter(maxp_p3)
    dotp1 = dot([flat_1, flat_p1], -1)
    dotp2 = dot([flat_1, flat_p2], -1)
    dotp3 = dot([flat_1, flat_p3], -1)

    shared_lstm = Bidirectional(LSTM(lstm_size, dropout=0.5, recurrent_dropout=0.2), merge_mode='concat', weights=None)
    encoded_a = shared_lstm(input_1)
    encoded_p1 = shared_lstm(input_p1)
    encoded_p2 = shared_lstm(input_p2)
    encoded_p3 = shared_lstm(input_p3)

    dotcnn_p1 = dot([encoded_p1, encoded_a], -1)
    dotcnn_p2 = dot([encoded_p2, encoded_a], -1)
    dotcnn_p3 = dot([encoded_p3, encoded_a], -1)

    drop = Dropout(rate=0.3)
    cnn_p1 = drop(concatenate([encoded_a, encoded_p1, dotcnn_p1], axis=-1))
    cnn_p2 = drop(concatenate([encoded_a, encoded_p2, dotcnn_p2], axis=-1))
    cnn_p3 = drop(concatenate([encoded_a, encoded_p3, dotcnn_p3], axis=-1))

    rnn_p1 = drop(concatenate([flat_1, flat_p1, dotp1], axis=-1))
    rnn_p2 = drop(concatenate([flat_1, flat_p2, dotp2], axis=-1))
    rnn_p3 = drop(concatenate([flat_1, flat_p3, dotp3], axis=-1))

    # merged_p1 = concatenate([rnn_p1, cnn_p1, manual_features_p1], axis=-1)
    # merged_p2 = concatenate([rnn_p2, cnn_p2, manual_features_p2], axis=-1)
    # merged_p3 = concatenate([rnn_p3, cnn_p3, manual_features_p3], axis=-1)

    merged_p1 = concatenate([rnn_p1, cnn_p1], axis=-1)
    merged_p2 = concatenate([rnn_p2, cnn_p2], axis=-1)
    merged_p3 = concatenate([rnn_p3, cnn_p3], axis=-1)
    dense = Dense(32, activation="relu")
    dense_p1 = dense(merged_p1)
    dense_p2 = dense(merged_p2)
    dense_p3 = dense(merged_p3)

    cont = concatenate([dense_p1, dense_p2, dense_p3])

    predictions = Dense(8, name='cnn_output', activation="softmax")(cont)

    model = Model([input_1, input_p1, input_p2, input_p3], predictions)
    # model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    batch_size = 128
    epochs = 2000
    max_length = 25
    filters = 32
    lstm_size = 64
    embedding_size = 300
    callbacks = []
    rnn_feature_path = "../paraphrasing-data/3para-features.pickle"
    mlp_feature_path = "../paraphrasing-data/3para-features.pickle"
    datasets = "../../paraphrasing-data/crowdsourced"

    X1, X2, X3, X4, Y, T = get_or_create_features_file(rnn_feature_path, datasets, max_length, False)
    # rnn_f = dict(zip(T, zip(X1, X2, Y)))
    # mX, mY, mT = mlp_features(mlp_feature_path, datasets, False)
    # mlp_f = dict(zip(mT, mX))

    # for key in rnn_f:
    #     rnn_f[key] = rnn_f[key] + (mlp_f[key],)
    #
    # T = list(rnn_f.keys())
    # X1, X2, Y, X = map(list, zip(*list(rnn_f.values())))

    z = list(zip(X1, X2, X3, X4, Y, T))
    random.shuffle(z)
    X1, X2, X3, X4, Y, T = zip(*z)

    T = np.array(T)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)

    _Y = Y
    Y = np.array(Y)
    Y = to_categorical(Y)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(_Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model(30, max_length, filters=filters, embedding_size=embedding_size, lstm_size=lstm_size)
            model.fit([X1[train], X2[train], X3[train], X4[train]], Y[train], batch_size=batch_size,
                      epochs=epochs, callbacks=callbacks,
                      verbose=0)

            scores = model.evaluate([X1[test], X2[test], X3[test], X4[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X1[test], X2[test], X3[test], X4[test]])
            y_pred = np.argmax(y_pred_v, axis=-1)
            # tn, fp, fn, tp = print_statistics(_Y[test], y_pred)
            # ttn += tn
            # tfp += fp
            # tfn += fn
            # ttp += tp
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            #
            # y_pred = y_pred.tolist()
            # y_test = Y[test]
            # t_test = T[test]
            #
            # q = queue.PriorityQueue()
            # for i in range(len(y_pred_v)):
            #     pred_class = y_pred[i]
            #     real_class = y_test[i] == 1
            #
            #     y = y_pred_v[i][0]
            #     y_r = y_test[i]
            #
            #     if pred_class[0] != real_class:
            #         q.put((- abs(y - y_r), y_r, y, t_test[i]))
            #
            #     if q.qsize() == 20:
            #         break
            #
            # while not q.empty():
            #     score, real, stimated, text = q.get()
            #     # print(real, stimated, text)  # features

        print("tn:", ttn, "fp:", tfp, "fn:", tfn, "tp:", ttp)
        print("Accuracy:", (ttp + ttn) / (ttp + ttn + tfp + tfn))
