from __future__ import print_function

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import model_from_json
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import multi_gpu_model, plot_model
from utils import *
import word2vec_client as word2vec
from dataset import para_vec_generator, corpus_generator

batch_size = 1024  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
max_input_seq_length = 15
max_target_seq_length = 15
datasets_directory = "/media/may/Data/ParaphrsingDatasets/all/"


# datasets_directory = "F:\ParaphrsingDatasets\\all/"



def define_models(input_we_size, output_we_size):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, input_we_size))  # num_encoder_tokens = 300
    encoder = LSTM(latent_dim, return_state=True, stateful=False)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, output_we_size))  # 300
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, stateful=False)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(output_we_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    single_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    try:
        # json_file = open('check-points/model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # True
        single_model.load_weights("check-points/epoch-82.hdf5")
    except:
        print('No saved model')

    multi_model = None
    # try:
    # It multi_gpu_model is not gonna work for se1-2-seq
    # It has a problem in cloning the model
    # multi_model = multi_gpu_model(single_model, gpus=2)
    # print("Training using multiple GPUs..")
    # except AssertionError as e:
    #     print(e)
    #     print("Training using single GPU or CPU..")

    if multi_model:
        model = multi_model
    else:
        model = single_model

    # with open("check-points/model.json", "w") as json_file:
    #     json_file.write(model.to_json())
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return single_model, model, encoder_model, decoder_model


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(np.array([input_seq]))

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, target_vector_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = START_OF_SENTENCE_VECTOR

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    decoded_sentence_tokens = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        # sampled_token_vector = np.argmax(output_tokens[0, -1, :])
        # sampled_word = reverse_target_word_index[sampled_token_vector]
        sampled_token_vector = output_tokens[0, -1, :]
        sampled_word = word2vec.most_similar_to_vector(sampled_token_vector.tolist())[0][0]
        decoded_sentence += " " + sampled_word
        decoded_sentence_tokens += 1
        # Exit condition: either hit max length or find stop token.
        if sampled_word == END_OF_SENTENCE_TOKEN or decoded_sentence_tokens > max_target_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, target_vector_size))
        target_seq[0, 0] = sampled_token_vector

        # Update states
        states_value = [h, c]

    return decoded_sentence


def sample_test():
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = TEST_SEQUENCE
    input_seq.append(END_OF_SENTENCE_VECTOR)
    # print(input_seq)
    decoded_sentence = decode_sequence(input_seq)
    print('**************************************************************')
    print('Input sentence:', TEST_TEXT)
    print('Decoded sentence:', decoded_sentence)
    print('**************************************************************')


class TestCallback(keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)
        if batch % 100 == 0:
            sample_test()


# Source/target Language samples
input_dataset = datasets_directory + "training.source.we"
target_dataset = datasets_directory + "training.target.we"
input_vector_size = 300
target_vector_size = 300

print('Number of samples:', -1)
print('Vector Size:', input_vector_size)
print('Max sequence length for inputs:', max_input_seq_length)
print('Max sequence length for outputs:', max_target_seq_length)

single_model, model, encoder_model, decoder_model = define_models(target_vector_size, input_vector_size)
filepath = "check-points/epoch-{epoch:02d}.hdf5"
callbacks_list = [
    TestCallback(),
    ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                    mode='auto', period=1),
    ModelCheckpoint("check-points/best-epoch.hdf5", monitor='val_loss', verbose=1, save_best_only=False,
                    save_weights_only=False, mode='auto', period=1)]

model.compile(optimizer='adam', loss='cosine_proximity', metrics=['acc'])
plot_model(model, to_file='check-points/model.png', show_shapes=True)

# generator = para_vec_generator(input_dataset, target_dataset,
#                                max_input_seq_length=max_input_seq_length,
#                                input_vector_size=input_vector_size,
#                                max_target_seq_length=max_target_seq_length,
#                                target_vector_size=target_vector_size,
#                                n=batch_size)
# 2170600

corpura = [
    "/media/may/Data/ParaphrsingDatasets/all/corpus/home.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/mrpc.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/MSR.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/multinli.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/P4P.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/quora.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/SLNI.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/sts.corpus",
    "/media/may/Data/ParaphrsingDatasets/all/corpus/TE.corpus",
    # "/media/may/Data/ParaphrsingDatasets/all/corpus/askubuntu.corpus",
]
generator = corpus_generator(corpura,
                             max_input_seq_length=max_input_seq_length,
                             input_vector_size=input_vector_size,
                             max_target_seq_length=max_target_seq_length,
                             target_vector_size=target_vector_size,
                             batch_size=batch_size)
# 2170600
steps_per_epoch = 2145484 / batch_size

model.fit_generator(generator,
                    epochs=epochs,
                    use_multiprocessing=False,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks_list)
sample_test()
# Save model
single_model.save('check-points/s2s.h5')  # Reverse-lookup token index to decode sequences back to

