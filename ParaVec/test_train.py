from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import model_from_json
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import multi_gpu_model, plot_model
import word2vec_client as word2vec
from dataset import read_dataset
from utils import *

batch_size = 1024  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
max_input_seq_length = 15
max_target_seq_length = 15
datasets_directory = "/media/may/Data/ParaphrsingDatasets/all/"


def define_models(num_encoder_tokens, num_decoder_tokens):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))  # num_encoder_tokens = 300
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))  # 3000
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    try:
        json_file = open('check-points/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("check-points/best-epoch.hdf5")
    except:
        print('No saved model')

    try:
        model = multi_gpu_model(model, cpu_relocation=True)
        print("Training using multiple GPUs..")
    except:
        print("Training using single GPU or CPU..")

    with open("check-points/model.json", "w") as json_file:
        json_file.write(model.to_json())
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

    return model, encoder_model, decoder_model


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


def create_dictionary(sentences):
    ret = set()
    for sentence in sentences:
        for w in sentence.strip().split(' '):
            ret.add(w)

    ret = sorted(list(ret))
    ret.append('<start>')
    return dict([(word, i) for i, word in enumerate(ret)])


# Source/target Language samples
input_dataset = "/media/may/Data/ParaphrsingDatasets/all/training.source.we"
target_dataset = "/media/may/Data/ParaphrsingDatasets/all/training.target.we"
input_vector_size = 300
target_vector_size = 300

print('Number of samples:', -1)
print('Vector Size:', input_vector_size)
print('Max sequence length for inputs:', max_input_seq_length)
print('Max sequence length for outputs:', max_target_seq_length)

model, encoder_model, decoder_model = define_models(target_vector_size, input_vector_size)
filepath = "check-points/epoch-{epoch:02d}.hdf5"
callbacks_list = [
    ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                    mode='auto', period=1),
    ModelCheckpoint("check-points/best-epoch.hdf5", monitor='val_loss', verbose=1, save_best_only=False,
                    save_weights_only=False, mode='auto', period=1)]
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='model.png', show_shapes=True)

for epoch in range(epochs):
    print("Epoch", epoch, "****************************************************")
    with open(input_dataset) as input_f, open(input_dataset) as target_f:
        while True:
            try:
                source_batch = list(next(read_dataset(input_f, batch_size)))
                target_batch = list(next(read_dataset(target_f, batch_size)))
            except:
                print("End of Epoch")
                break

            assert len(source_batch) == len(target_batch)
            if len(source_batch) == 0:
                break

            # The last batch might be less than batch_size
            real_batch_size = len(source_batch)

            encoder_input_data = np.zeros((real_batch_size, max_input_seq_length, input_vector_size), dtype='float32')
            decoder_input_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size), dtype='float32')
            decoder_target_data = np.zeros((real_batch_size, max_target_seq_length, target_vector_size),
                                           dtype='float32')

            for i, (input_text, target_text) in enumerate(zip(source_batch, target_batch)):
                for t, word in enumerate(input_text):
                    if t < max_input_seq_length:
                        encoder_input_data[i, t] = word
                for t, word in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    if t < max_input_seq_length:
                        decoder_input_data[i, t] = word
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            decoder_target_data[i, t - 1] = word

            model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                batch_size=batch_size,
                                epochs=1,
                                # use_multiprocessing=True,
                                callbacks=callbacks_list,
                                validation_split=0.2)
            sample_test()
            model.save('check-points/s2s.h5')
            # Save model


# Reverse-lookup token index to decode sequences back to
# something readable.
# reverse_input_word_index = dict(
#     (i, word) for word, i in input_token_index.items())
# reverse_target_word_index = dict(
#     (i, word) for word, i in target_token_index.items())
