from __future__ import print_function
from audio_preprocessing.cconfig import config
from keras.models import model_from_json
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_model(l_model_name, print_sum=False):

    model_input_file = config.datapath + '/weight_matrices/' + l_model_name + "_model.json"
    model_weights_input_file = config.datapath + '/weight_matrices/' + l_model_name + "_weights.h5"
    json_file = open(model_input_file, 'r')
    loaded_model_json = json.load(json_file)
    l_model = model_from_json(loaded_model_json)
    # load model weights
    l_model.load_weights(model_weights_input_file)
    if print_sum:
        l_model.summary()
    return l_model


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def print_some_weights(model):
    # get first conv layer and first weigth matrix which hold W weights

    for idx, m_layer in enumerate(model.layers):

        print("layer name ", m_layer.name)
        if m_layer.name == '1_timedist':
            c_weights = m_layer.get_weights()[0]
            print("layer name ", m_layer.name)
            print("c_weights.shape ", c_weights.shape)
            c_w_mean = np.mean(c_weights)
            c_w_stddev = np.std(c_weights)
            print("mean, stddev ", c_w_mean, c_w_stddev)
            plot_hist_weights(c_weights.flatten(), 100)

        if m_layer.name == 'dense_b_lstm':
            c_weights = m_layer.get_weights()[0]
            print("layer name ", m_layer.name)
            print("c_weights.shape ", c_weights.shape)
            c_w_mean = np.mean(c_weights)
            c_w_stddev = np.std(c_weights)
            print("mean, stddev ", c_w_mean, c_w_stddev)
            plot_hist_weights(c_weights.flatten(), 100)


        if m_layer.name == 'lstm_1':
            # weights of LSTM layer
            lstm_c_weights = m_layer.get_weights()[0]
            lstm_f_weights = m_layer.get_weights()[3]
            lstm_i_weights = m_layer.get_weights()[6]
            lstm_o_weights = m_layer.get_weights()[9]
            print("lstm_c_weights.shape ", lstm_c_weights.shape)
            print("lstm_f_weights.shape ", lstm_f_weights.shape)
            print("lstm_i_weights.shape ", lstm_i_weights.shape)
            print("lstm_o_weights.shape ", lstm_o_weights.shape)
            print("lstm_c_weights: mean, stddev ", np.mean(lstm_c_weights), np.std(lstm_c_weights))
            print("lstm_f_weights: mean, stddev ", np.mean(lstm_f_weights), np.std(lstm_f_weights))
            print("lstm_i_weights: mean, stddev ", np.mean(lstm_i_weights), np.std(lstm_i_weights))
            print("lstm_o_weights: mean, stddev ", np.mean(lstm_o_weights), np.std(lstm_o_weights))

            plt.subplot(4,1,1)
            n, bins, patches = plt.hist(lstm_c_weights.reshape(lstm_c_weights.shape[0] * lstm_c_weights.shape[1]), 100)
            plt.subplot(4, 1, 2)
            n, bins, patches = plt.hist(lstm_f_weights.reshape(lstm_f_weights.shape[0] * lstm_f_weights.shape[1]), 100)
            plt.subplot(4, 1, 3)
            n, bins, patches = plt.hist(lstm_i_weights.reshape(lstm_i_weights.shape[0] * lstm_i_weights.shape[1]), 100)
            plt.subplot(4, 1, 4)
            n, bins, patches = plt.hist(lstm_o_weights.reshape(lstm_o_weights.shape[0] * lstm_o_weights.shape[1]), 100)
            plt.show()


def plot_hist_weights(weights, bins=10):

    n, bins, patches = plt.hist(weights, bins=bins)
    plt.show()


import_dir = config.datapath + "weight_matrices/"
model_name = 'guitar_train_4files_5sec_60res_1400maxf_m2_512hid_1lyrs_100ep_linearact'
# model_name = 'guitar_train_45files_5sec_60res_1400maxf_m1_512hid_1lyrs_180ep_linearact'
w_filename = model_name + "_weights.h5"
abs_path_to_file = import_dir + w_filename
print_structure(abs_path_to_file)
model = get_model(model_name, True)
print_some_weights(model)