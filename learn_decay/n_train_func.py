from __future__ import print_function
import os.path
import json
from audio_preprocessing.cconfig import config
from audio_preprocessing.pipeline import load_matrix, AudioPipeline
from learn_decay.lstm_utils import create_lstm_network
from learn_decay.CRNN import create_conv_lstm_network
import numpy as np


def train_func(train_dir, model_type='LRNN', matrix_file='', n_hid=1024, n_recur=1, epochs=100, batch_size=10,
               n_to_load=1, highest_freq=5000, clip_len=2, mat_dirs=None, chunks_per_sec=4,
               down_sampling=False, root_to_folder='/instrument_samples/', save_weights=True, add_spectra=False):

    if matrix_file is '':
        matrix_file = train_dir
    dpath = config.datapath + root_to_folder + train_dir
    fpath = dpath + '/' + matrix_file + '.npy'

    d_mat_name = '/' + matrix_file + '_' + str(n_to_load) + 'files'
    if down_sampling:
        d_mat_name += str(chunks_per_sec) + 'res'
        d_mat_name += str(highest_freq) + 'maxf'
    else:
        d_mat_name += '_raw'
    if add_spectra:
        d_mat_name += '_spec'

    # see if matrix file exists
    if not os.path.isfile(fpath):
        # if not, create it from data
        dpath = config.datapath + root_to_folder + train_dir
        if os.path.isdir(dpath):
            audios = AudioPipeline((root_to_folder + train_dir), n_to_load=n_to_load, highest_freq=highest_freq,
                                   clip_len=clip_len, mat_dirs=mat_dirs, chunks_per_sec=chunks_per_sec,
                                   down_sampling=down_sampling, add_spectra=add_spectra)

            audios.create_train_matrix(f_name_out=d_mat_name)
        else:
            print('both entered paths are invalid. No data loaded')
            print('train directory: ', dpath)
            print('matrix file: ', fpath)
            return

        data = load_matrix(root_to_folder + train_dir + '/', d_mat_name)
    else:
        data = load_matrix(root_to_folder + train_dir + '/', matrix_file)

    x_data = data['x_data']
    y_data = data['y_data']

    num_frequency_dimensions = x_data.shape[2]

    # create model
    if model_type == 'LRNN':
        model = create_lstm_network(num_frequency_dimensions, n_hid, n_recur)
    elif model_type == 'CRNN':
        y_data = x_data[:,:,:]
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))

        model = create_conv_lstm_network(num_frequency_dimensions, n_hid, n_recur)

    model.summary()
    w_mat_name = ""
    print('Start Training')
    model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.0)

    print('Training complete')
    w_mat_name = d_mat_name + '_' + str(n_hid) + 'hid_' + str(epochs) + 'ep'

    if save_weights:
        json_string = model.to_json()
        model_output = config.datapath + '/weight_matrices/' + w_mat_name + '_model.json'
        fout = open(model_output, 'w')
        json.dump(json_string, fout)
        fout.close()
        print('saved model to: ' + model_output)

        weights_output = config.datapath + '/weight_matrices/' + w_mat_name + '_weights.h5'
        model.save_weights(weights_output, overwrite=True)
        print('saved weights to: ' + weights_output)

    return model, w_mat_name
