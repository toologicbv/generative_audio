from __future__ import print_function
import numpy as np
import tensorflow as tf
from train_func import train_func


def flute_train():
    train_dir = 'flute_nonvib_train'
    print("Using flute nonvib training set ")
    _, w_mat_name, d_mat_name = train_func(train_dir,
                                architecture='1',
                                n_hid_neurons=1024,
                                n_rec_layers=1,
                                epochs=50,
                                highest_freq=8000,
                                n_to_load=39,
                                down_sampling=True,
                                save_weights=True,
                                chunks_per_sec=60,
                                clip_len=3,
                                activation='linear')

    print("Saved weights to %s" % w_mat_name)


def piano_train():
    train_dir = 'piano_train'
    print("Using piano_training set ")
    _, w_mat_name, d_mat_name = train_func(train_dir,
                                architecture='1',
                                n_hid_neurons=1024,
                                n_rec_layers=1,
                                epochs=50,
                                highest_freq=4200,
                                n_to_load=88,
                                down_sampling=True,
                                save_weights=True,
                                chunks_per_sec=60,
                                clip_len=5,
                                activation='linear')

    print("Saved weights to %s" % w_mat_name)


def cello_pizz_train():
    train_dir = 'cello_pizz_train'
    print("Using cello_pizz training set ")
    _, w_mat_name, d_mat_name = train_func(train_dir,
                                batch_size=100,
                                architecture='1',
                                n_hid_neurons=1024,
                                n_rec_layers=1,
                                epochs=500,
                                highest_freq=5000,
                                n_to_load=90,
                                down_sampling=True,
                                save_weights=True,
                                chunks_per_sec=60,
                                clip_len=3,
                                activation='linear')

    print("Saved weights to %s" % w_mat_name)


def guitar_train():

    print("Using guitar training set ")
    train_dir = 'guitar_train'

    _, w_mat_name, d_mat_name = train_func(train_dir,
                                batch_size=100,
                                architecture='2',
                                n_hid_neurons=512,
                                n_rec_layers=1,
                                epochs=100,
                                highest_freq=1400,
                                n_to_load=4,
                                down_sampling=True,
                                save_weights=True,
                                chunks_per_sec=60,
                                clip_len=5,
                                activation='linear')

    print("Saved weights to %s %s" % (w_mat_name, d_mat_name))

guitar_train()