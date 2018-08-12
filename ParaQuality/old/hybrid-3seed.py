import os
import pickle
import queue
import random

import numpy as np
import tensorflow as tf
from ParaQuality.mlp import print_statistics
from ParaQuality.rnn import sentence_to_vec
from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPool1D, Flatten, dot, Dropout, Bidirectional, LSTM, average
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.features import OmniFeatureFunction, Sentence2VecSimilarityFF
from utils.dataset import read_paraphrased_tsv_files
from utils.preprocess import remove_marks


def create_model(manual_features=31, max_length=25, filters=32, embedding_size=300, lstm_size=64, print_summary=False):
    input_p = Input(shape=(max_length, embedding_size))

    input_f_1 = Input(shape=(manual_features,))
    input_f_2 = Input(shape=(manual_features,))
    input_f_3 = Input(shape=(manual_features,))

    input_s_1 = Input(shape=(max_length, embedding_size))
    input_s_2 = Input(shape=(max_length, embedding_size))
    input_s_3 = Input(shape=(max_length, embedding_size))

    conv = Conv1D(filters=filters, kernel_size=4)
    conv_out_p = conv(input_p)
    conv_out_s1 = conv(input_s_1)
    conv_out_s2 = conv(input_s_2)
    conv_out_s3 = conv(input_s_3)

    max_layer = MaxPool1D(pool_size=4)
    max_p = max_layer(conv_out_p)
    max_s1 = max_layer(conv_out_s1)
    max_s2 = max_layer(conv_out_s2)
    max_s3 = max_layer(conv_out_s3)

    #
    flatter = Flatten()
    flat_p, flat_s = flatter(max_p), average([flatter(max_s1), flatter(max_s2), flatter(max_s3)])
    dotcnn = dot([flat_p, flat_s], -1)

    shared_lstm = Bidirectional(LSTM(lstm_size, dropout=0.5, recurrent_dropout=0.3), merge_mode='concat', weights=None)
    lstm_p = shared_lstm(input_p)
    lstm_s1 = shared_lstm(input_s_1)
    lstm_s2 = shared_lstm(input_s_2)
    lstm_s3 = shared_lstm(input_s_3)
    lstm_s = average([lstm_s1, lstm_s2, lstm_s3])
    dotrnn = dot([lstm_s, lstm_p], -1)

    drop = Dropout(rate=0.3)
    rnn = drop(concatenate([lstm_p, lstm_s, dotrnn], axis=-1))
    cnn = drop(concatenate([flat_p, flat_s, dotcnn], axis=-1))

    manual_features = average([input_f_1, input_f_2, input_f_3])

    merged = concatenate([rnn, cnn, manual_features], axis=-1)
    dense1 = Dense(32, activation="softmax")(merged)

    predictions = Dense(1, name='output', activation="sigmoid")(dense1)
    # predictions = Activation("sigmoid")(dense)

    model = Model([input_p, input_f_1, input_f_2, input_f_3, input_s_1, input_s_2, input_s_3], predictions)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


def find_diverse_seeds(expression, samples):
    seeds = [expression]
    samples = [(i + 1, s) for i, s in enumerate(samples) if s[0] not in seeds]
    sentvec = Sentence2VecSimilarityFF()

    less_similar = 1
    less_similar_sample = None
    for s in samples:
        if s[1][1] != 'valid':
            continue

        v = sentvec.extract(expression, s[1][0], 0)
        if v < less_similar:
            less_similar = v
            less_similar_sample = s[1][0]

    if less_similar_sample:
        seeds.append(less_similar_sample)

    less_similar = 2
    less_similar_sample = None
    for s in samples:
        if s[1][1] != 'valid':
            continue

        v1 = sentvec.extract(expression, s[1][0], 0)
        v2 = sentvec.extract(seeds[1], s[1][0], 0)
        if v1 + v2 < less_similar:
            less_similar = v1 + v2
            less_similar_sample = s[1][0]

    if less_similar_sample:
        seeds.append(less_similar_sample)

    return seeds, [s for s in samples if s[1][0] not in seeds]


def to_vector(dataset, max_length, save):
    # vectorize
    Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = [], [], [], [], [], [], [], [], []

    ffs = OmniFeatureFunction()
    for i, expression in enumerate(dataset):
        seeds, samples = find_diverse_seeds(expression, dataset[expression])

        expression_1_vec = sentence_to_vec(seeds[0], max_length)
        expression_2_vec = sentence_to_vec(seeds[1], max_length)
        expression_3_vec = sentence_to_vec(seeds[2], max_length)
        index = 0
        for instance in samples:
            index += 1
            paraphrase = instance[1][0]
            paraphrase_vec = sentence_to_vec(paraphrase, max_length)
            f1 = ffs.extract(seeds[0], paraphrase, instance[0])
            f2 = ffs.extract(seeds[1], paraphrase, instance[0])
            f3 = ffs.extract(seeds[2], paraphrase, instance[0])
            Xf1.append(f1)
            Xf2.append(f2)
            Xf3.append(f3)
            Xs1.append(expression_1_vec)
            Xs2.append(expression_2_vec)
            Xs3.append(expression_3_vec)
            Xp.append(paraphrase_vec)
            Y.append(0 if instance[1][1] == 'invalid' else 1)
            T.append(expression + "<==>" + seeds[0] + "<==>" + seeds[1] + "<==>" + seeds[2])
        print("Dataset Processed", int((i + 1) / len(dataset) * 100), "%")
    if save:
        pickle.dump((Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T), open(save, 'wb'))

    return np.array(Xp), np.array(Xf1), np.array(Xf2), np.array(Xf3), np.array(Xs1), np.array(Xs2), np.array(
        Xs3), np.array(Y), np.array(T)


def get_or_create_features_file(dataset_feature_path, datasets_path, max_length=25, shuffle=True):
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = pickle.load(f)
    else:
        dataset = read_paraphrased_tsv_files(datasets_path, processor=remove_marks)
        Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = to_vector(dataset, max_length, save=dataset_feature_path)

    if shuffle:
        z = list(zip(Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T))
        random.shuffle(z)
        Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = zip(*z)

    Xp = np.array(Xp)
    Xf1 = np.array(Xf1)
    Xf2 = np.array(Xf2)
    Xf3 = np.array(Xf3)
    Xs1 = np.array(Xs1)
    Xs2 = np.array(Xs2)
    Xs3 = np.array(Xs3)
    Y = np.array(Y)
    T = np.array(T)

    print("Input :", Xp.shape, Xs1.shape, Xf1.shape)
    print("Output :", Y.shape)
    return Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T


if __name__ == "__main__":

    batch_size = 128  # the more is the size, the less fn
    epochs = 75
    max_length = 25
    filters = 32
    lstm_size = 64
    embedding_size = 300
    callbacks = []

    datasets = "../paraphrasing-data/crowdsourced"
    feature_file = "../paraphrasing-data/hybrid_3seeds_feature_file.pickle"

    Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = get_or_create_features_file(feature_file, datasets, max_length, False)
    z = list(zip(Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T))
    random.shuffle(z)
    Xp, Xf1, Xf2, Xf3, Xs1, Xs2, Xs3, Y, T = zip(*z)

    Xp = np.array(Xp)
    Xf1 = np.array(Xf1)
    Xf2 = np.array(Xf2)
    Xf3 = np.array(Xf3)
    Xs1 = np.array(Xs1)
    Xs2 = np.array(Xs2)
    Xs3 = np.array(Xs3)
    Y = np.array(Y)
    T = np.array(T)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model(31, max_length, filters=filters, embedding_size=embedding_size, lstm_size=lstm_size)
            model.fit([Xp[train], Xf1[train], Xf2[train], Xf3[train], Xs1[train], Xs2[train], Xs3[train]], Y[train],
                      batch_size=batch_size,
                      epochs=epochs, callbacks=callbacks,
                      verbose=0)
            scores = model.evaluate([Xp[test], Xf1[test], Xf2[test], Xf3[test], Xs1[test], Xs2[test], Xs3[test]],
                                    Y[test], verbose=1)
            y_pred_v = model.predict([Xp[test], Xf1[test], Xf2[test], Xf3[test], Xs1[test], Xs2[test], Xs3[test]])
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            y_pred = (y_pred_v >= 0.3)
            tn, fp, fn, tp = print_statistics(Y[test], y_pred)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
            print("Threshold  > Acc: %.2f%%" % (100 * (tp + tn) / (tp + tn + fn + fp)))

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
