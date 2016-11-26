from generate_seq import *


# parameters for generation experiment
folder_spec = 'instrument_samples/cello_arco_train/'
data = 'cello_train_86files_10res_8000maxf'
model_name = 'cello_train_86files_10res_8000maxf_1024hid_200ep'

# folder_spec = 'instrument_samples/cello_pizz_train/'
# data = 'cello_pizz_train_90files10res8000maxf'
# model_name = 'cello_pizz_train_90files10res8000maxf_1024hid_50ep'

folder_spec = 'instrument_samples/guitar_train/'
data = 'guitar_train_45files_5sec_60res_1400maxf'
model_name = 'guitar_train_45files_5sec_60res_1400maxf_m1_512hid_1lyrs_180ep_tanhact'

folder_spec = 'instrument_samples/guitar_train/'
data = 'guitar_train_45files_5sec_60res_1400maxf'
model_name = 'guitar_train_45files_5sec_60res_1400maxf_m1_512hid_1lyrs_180ep_linearact'

# folder_spec = 'instrument_samples/piano_train/'
# data = 'piano_train_37files_5sec_60res_4200maxf'
# model_name = 'piano_train_37files_5sec_60res_4200maxf_m1_512hid_1lyrs_180ep_linearact'


prime_length = 20
num_of_tests = 2

gen_seq_full(folder_spec=folder_spec, data=data, model_name=model_name,
             prime_length=prime_length, num_of_tests=num_of_tests, architecture='1',
             mean_std_per_file=False)

