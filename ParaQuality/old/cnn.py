import queue

import tensorflow as tf
from ParaQuality.mlp import print_statistics, reweight
from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPool1D, Flatten, Activation, dot, Dropout
from sklearn.cross_validation import StratifiedKFold

from ParaQuality.old.rnn import get_or_create_features_file


def create_model(shape, filters, print_summary=False, compile=True):
    input_1 = Input(shape=shape)
    input_2 = Input(shape=shape)

    conv = Conv1D(filters=filters, kernel_size=4)

    conv_out_1 = conv(input_1)
    conv_out_2 = conv(input_2)

    maxp_1 = MaxPool1D(pool_size=6)(conv_out_1)

    maxp_2 = MaxPool1D(pool_size=6)(conv_out_2)

    conv_2 = Conv1D(filters=int(filters / 2), kernel_size=2)

    conv2_out_1 = conv_2(maxp_1)
    conv2_out_2 = conv_2(maxp_2)

    maxp2_1 = MaxPool1D(pool_size=2)(conv2_out_1)
    flat_1 = Flatten()(maxp2_1)

    maxp2_2 = MaxPool1D(pool_size=2)(conv2_out_2)
    flat_2 = Flatten()(maxp2_2)

    dotp = dot([flat_1, flat_2], -1)
    merged = concatenate([flat_1, flat_2, dotp], axis=-1)
    dropped = Dropout(rate=0.3)(merged)
    dense1 = Dense(32)(dropped)

    dense = Dense(1, name='cnn_output')(dense1)
    output = Activation("sigmoid")(dense)

    if not compile:
        return input_1, input_2, output

    model = Model([input_1, input_2], output)
    model.compile(loss=reweight, optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    batch_size = 128
    epochs = 75
    max_length = 25
    filters = 32
    embedding_size = 300
    callbacks = []
    dataset_feature_path = "../paraphrasing-data/rnn-features.pickle"
    datasets = "../paraphrasing-data/crowdsourced"

    X1, X2, Y, T = get_or_create_features_file(dataset_feature_path, datasets, max_length)

    with tf.device('/gpu:1'):
        kfold = StratifiedKFold(Y, n_folds=5)
        ttn, tfp, tfn, ttp = 0, 0, 0, 0
        for train, test in kfold:
            model = create_model((max_length, embedding_size), filters=filters)
            model.fit([X1[train], X2[train]], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                      verbose=0)
            scores = model.evaluate([X1[test], X2[test]], Y[test], verbose=1)
            y_pred_v = model.predict([X1[test], X2[test]])
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
