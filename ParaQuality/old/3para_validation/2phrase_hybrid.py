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


def create_prediction_model():
    input = Input(shape=(3,))
    dense1 = Dense(16, activation="relu")(input)
    predictions = Dense(8, activation="softmax")(dense1)

    model = Model(input, predictions)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model


def create_similarity_model(manual_features=30, max_length=25, filters=32, embedding_size=300, lstm_size=64,
                            print_summary=False):
    manual_features = Input(shape=(manual_features,))
    input_1 = Input(shape=(max_length, embedding_size))
    input_2 = Input(shape=(max_length, embedding_size))

    conv = Conv1D(filters=filters, kernel_size=4)
    conv_out_1 = conv(input_1)
    conv_out_2 = conv(input_2)

    max_layer = MaxPool1D(pool_size=4)
    maxp_1 = max_layer(conv_out_1)
    maxp_2 = max_layer(conv_out_2)

    flatter = Flatten()
    flat_1, flat_2 = flatter(maxp_1), flatter(maxp_2)
    dotcnn = dot([flat_1, flat_2], -1)

    shared_lstm = Bidirectional(LSTM(lstm_size, dropout=0.5, recurrent_dropout=0.3), merge_mode='concat', weights=None)
    encoded_a = shared_lstm(input_1)
    encoded_b = shared_lstm(input_2)
    dotrnn = dot([encoded_b, encoded_a], -1)

    drop = Dropout(rate=0.3)
    cnn = drop(concatenate([encoded_a, encoded_b, dotcnn], axis=-1))
    rnn = drop(concatenate([flat_1, flat_2, dotrnn], axis=-1))

    merged = concatenate([rnn, cnn, manual_features], axis=-1)
    dense1 = Dense(32, activation="softmax")(merged)

    predictions = Dense(1, name='cnn_output', activation="sigmoid")(dense1)
    # predictions = Activation("sigmoid")(dense)

    model = Model([manual_features, input_1, input_2], predictions)
    # model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


get_bin = lambda x, n: format(x, 'b').zfill(n)


def merge_inputs(X1, X2, X3, X4, Y):
    ret_X1 = []
    ret_X2 = []
    ret_Y = []

    for i in range(len(X1)):
        for k in range(3):
            ret_X1.append(X1[i])

        for y in get_bin(np.argmax(Y[i], -1), 3):
            ret_Y.append(int(y))

        ret_X2.append(X2[i])
        ret_X2.append(X3[i])
        ret_X2.append(X4[i])

    return np.zeros((len(ret_X1), 30)), np.array(ret_X1), np.array(ret_X2), np.array(ret_Y)


def predict(model, X2, X3, X4, Y):
    preds = []
    zf = np.zeros((1, 30))
    for i in range(len(X2)):
        y1 = model.predict([zf, np.array([X1[i]]), np.array([X2[i]])])[0][0]
        y2 = model.predict([zf, np.array([X1[i]]), np.array([X3[i]])])[0][0]
        y3 = model.predict([zf, np.array([X1[i]]), np.array([X4[i]])])[0][0]

        preds.append([y1, y2, y3])

    return np.array(preds)


if __name__ == "__main__":

    batch_size = 128
    epochs = 75
    max_length = 25
    filters = 32
    lstm_size = 64
    embedding_size = 300
    callbacks = []
    rnn_feature_path = "../../paraphrasing-data/3para-features.pickle"
    mlp_feature_path = "../../paraphrasing-data/3para-features.pickle"
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
        fold = 0
        with open("ParaQualityHybrid.arff", "wt") as f:
            f.write(
                "@relation ParaQuality\n@attribute attr_1 numeric\n@attribute attr_2 numeric\n@attribute attr_3 "
                "numeric\n@attribute score {0,1,2,3,4,5,6,7}\n@data\n")

            for train, test in kfold:
                # start training
                model = create_similarity_model(30, max_length, filters=filters, embedding_size=embedding_size,
                                                lstm_size=lstm_size)

                manual_features, single_X1, single_X2, single_Y = merge_inputs(X1[train], X2[train], X3[train],
                                                                               X4[train],
                                                                               Y[train])

                z = list(zip(manual_features, single_X1, single_X2, single_Y))
                random.shuffle(z)
                manual_features, single_X1, single_X2, single_Y = zip(*z)
                manual_features = np.array(manual_features)
                single_X1 = np.array(single_X1)
                single_X2 = np.array(single_X2)
                single_Y = np.array(single_Y)

                model.fit([manual_features, single_X1, single_X2], single_Y, batch_size=batch_size,
                          epochs=epochs, callbacks=callbacks,
                          verbose=0)

                # model_predictions = predict(model, X1[train], X2[train], X3[train], X4[train])
                model_predictions = predict(model, X1[test], X2[test], X3[test], X4[test])

                # final_model = create_prediction_model()

                for i in range(len(model_predictions)):
                    f.write(",".join([str(j>0.5) for j in model_predictions[i]]))
                    f.write("," + str(np.argmax(Y[test][i], -1)) + "\n")
                print("Saved")
                # final_model.fit([model_predictions], Y[train], batch_size=128,
                #                 epochs=300, callbacks=callbacks,
                #                 verbose=0)

                # Start Testing
                # t_manual_features, t_single_X1, t_single_X2, t_single_Y = merge_inputs(X1[test], X2[test], X3[test],
                #                                                                        X4[test],
                #                                                                        Y[test])
                #
                # t_model_predictions = predict(model, X1[test], X2[test], X3[test], X4[test])
                # final_prediction = final_model.predict([t_model_predictions])
                # scores = final_model.evaluate([t_model_predictions], Y[test])
                # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

                # y_pred_v = model.predict([t_model_predictions])
                # y_pred = np.argmax(y_pred_v, axis=-1)
