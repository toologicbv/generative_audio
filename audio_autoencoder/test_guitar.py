from __future__ import print_function
from audio_preprocessing.pipeline import AudioPipeline
from ConvAutoencoder import ConvAutoencoder
import matplotlib.pyplot as plt

import theano.tensor as T
import theano

root_to_folder = 'instrument_samples/'
train_dir = root_to_folder + 'guitar_train/'
data = train_dir + 'sines_mat'
audios = AudioPipeline(train_dir, n_to_load=4, highest_freq=1400,
                           clip_len=5, mat_dirs=None, chunks_per_sec=60,
                           down_sampling=True)

batches = audios.train_batches()
x_data = next(batches).divisible_matrix(16)
x_data = next(batches).divisible_matrix(16)
print(x_data.shape)
x_train = x_data[:, :]
# x_test = x_train[:, :]

x_test = x_data[6:13, :]
# plt.plot(x_train.reshape(x_train.shape[0] * x_train.shape[1]))
# plt.show()

#
auto = ConvAutoencoder(x_train, x_test)
# #
auto.train(100, 10, True)
# #
gen_test_signal = auto.test_encoder
print("gen_test_signal.shape ", gen_test_signal.shape)
t = gen_test_signal[:, :]
plt.subplot(2, 1, 1)
plt.plot(t.reshape(t.shape[0] * t.shape[1]))
plt.subplot(2, 1, 2)
t = x_train[:, :]
plt.plot(t.reshape(t.shape[0] * t.shape[1]))
plt.show()