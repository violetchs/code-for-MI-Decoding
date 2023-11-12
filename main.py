import h5py
import os
import numpy as np
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
import matplotlib.pyplot as plt
from utils import spike_toarray, trail_split, divide_trail_by_orientation, divide_spike_by_orientation, \
    plot_raster, plot_psth, plot_tuningcurve, divide_spike_by_kinematics, plot_tuning, neuron_encoding_cursor, \
    plot_r2, KalmanFilterRegression, get_R2, get_rho, LSTMRegression

if __name__ == '__main__':

    data_root = './3854034'
    file_name = 'indy_20160921_01.mat'
    data_path = os.path.join(data_root, file_name)
    Data = h5py.File(data_path, 'r')
    chan_names = Data['chan_names']
    cursor_pos = np.array(Data['cursor_pos'])
    finger_pos = np.array(Data['finger_pos'])
    spikes = Data['spikes']
    t = np.squeeze(np.array(Data['t']))
    target_pos = np.array(Data['target_pos'])
    wf = np.array(Data['wf'])

    nchannel = chan_names.shape[-1]
    npoints = t.shape[-1]

    spike_neuron, neuron_num_perch = spike_toarray(nchannel, spikes, Data)
    nneuron = int(sum(neuron_num_perch))
    trails_info = trail_split(target_pos, npoints)

    trail_divided = divide_trail_by_orientation(trails_info)

    spike_devided, t_divided = divide_spike_by_orientation(trails_info, trail_divided, npoints, t, spike_neuron, neuron_num_perch)

    #plot_raster(spike_devided, t_divided)
    #plot_psth(spike_devided, )
    #plot_tuningcurve(spike_devided, t_divided)
    #neuron_encoding_cursor(finger_pos, spike_neuron, neuron_num_perch, t, npoints)
    bin=0.1
    overlap = 0
    kinematics, spike_samples = divide_spike_by_kinematics(finger_pos, bin, overlap, npoints, t, spike_neuron, neuron_num_perch, kinematic='pos')


    pos_binned = np.array(kinematics)
    temp = np.diff(pos_binned, axis=0)
    vels_binned = np.concatenate((temp, temp[-1:, :]), axis=0)
    temp2 = np.diff(vels_binned, axis=0)
    acc_binned = np.concatenate((temp2, temp2[-1:, :]), axis=0)
    y_kf = np.concatenate((pos_binned, vels_binned, acc_binned), axis=1)
    #y_kf = np.concatenate((pos_binned, vels_binned), axis=1)
    #y_kf = acc_binned
    x_kf = np.transpose(np.array(spike_samples))

    lag = -1#lag=-1 means use the spikes 1 bin before the output)
    y_kf = y_kf[-lag:, :]
    x_kf = x_kf[:lag, :]

    training_range = [0, 0.85]
    valid_range = [0.85, 1]
    testing_range = [0.85, 1]

    num_examples_kf = x_kf.shape[0]

    # Note that each range has a buffer of 1 bin at the beginning and end
    # This makes it so that the different sets don't include overlapping data
    training_set = np.arange(int(np.round(training_range[0] * num_examples_kf)) + 1,
                             int(np.round(training_range[1] * num_examples_kf)) - 1)
    testing_set = np.arange(int(np.round(testing_range[0] * num_examples_kf)) + 1,
                            int(np.round(testing_range[1] * num_examples_kf)) - 1)
    valid_set = np.arange(int(np.round(valid_range[0] * num_examples_kf)) + 1,
                          int(np.round(valid_range[1] * num_examples_kf)) - 1)

    # Get training data
    X_kf_train = x_kf[training_set, :]
    y_kf_train = y_kf[training_set, :]

    # Get testing data
    X_kf_test = x_kf[testing_set, :]
    y_kf_test = y_kf[testing_set, :]

    # Get validation data
    X_kf_valid = x_kf[valid_set, :]
    y_kf_valid = y_kf[valid_set, :]

    X_kf_train_mean = np.nanmean(X_kf_train, axis=0)
    X_kf_train_std = np.nanstd(X_kf_train, axis=0)
    X_kf_train = (X_kf_train - X_kf_train_mean) / X_kf_train_std
    X_kf_test = (X_kf_test - X_kf_train_mean) / X_kf_train_std
    X_kf_valid = (X_kf_valid - X_kf_train_mean) / X_kf_train_std

    # Zero-center outputs
    y_kf_train_mean = np.mean(y_kf_train, axis=0)
    y_kf_train = y_kf_train - y_kf_train_mean
    y_kf_test = y_kf_test - y_kf_train_mean
    y_kf_valid = y_kf_valid - y_kf_train_mean

    # Declare model
    c = [1]
    for i in c:
        for j in c:
            model_kf = KalmanFilterRegression(C1=i, C2=j)  # There is one optional parameter (see ReadMe)
            model_kf.fit(X_kf_train, y_kf_train)
            y_valid_predicted_kf = model_kf.predict(X_kf_valid, y_kf_valid)

            rho_kf = get_rho(y_kf_valid, y_valid_predicted_kf)
            print('C1_{}_C2_{}_rho2:'.format(i, j),
                  rho_kf[0:2] ** 2)

    '''[A, W, H, Q] = model_kf.model
    z = (y_kf - y_kf_train_mean) * H.T
    z_true = x_kf
    z = np.array(z)
    z = z * X_kf_train_std + X_kf_train_mean
    z[np.where(z<0)] = 0
    z_true = np.array(z_true)
    R2_1 = get_R2(z_true, z)
    #for i in range(R2.shape[0]):
        #print('{}_r2:{}'.format(i, R2[i]))
    # Get predictions'''
    y_valid_predicted_kf = model_kf.predict(X_kf_valid, y_kf_valid)


    # Get metrics of fit (see read me for more details on the differences between metrics)
    # First I'll get the R^2
    #R2_kf = get_R2(y_kf_valid, y_valid_predicted_kf)
    #print('R2:', R2_kf[0:2])  # I'm just printing the R^2's of the 1st and 2nd entries that correspond to the positions
    # Next I'll get the rho^2 (the pearson correlation squared)

    rho_kf = get_rho(y_kf_valid, y_valid_predicted_kf)
    print('C1rho2:',
          rho_kf[0:2] ** 2)  #printing the rho^2's of the 1st and 2nd entries that correspond to the positions
    x = [i for i in range(0, 538, 50)]
    xlabel = np.array([5 * i for i in range(len(x))])
    plt.plot(y_kf_valid[:, 1] + y_kf_train_mean[1], 'black', label='True')
    plt.plot(y_valid_predicted_kf[:, 1] + y_kf_train_mean[1], 'r', label='Predicted')
    plt.xlabel('Time/s')
    plt.ylabel('Position/cm')
    plt.title('py')
    plt.xticks(x, xlabel)
    plt.legend()
    #plt.show()

    print(spikes[0, 0])
    bins_before = 7  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 0  # How many bins of neural data after the output are used for decoding
    X = get_spikes_with_history(np.transpose(np.array(spike_samples)), bins_before, bins_after, bins_current)
    X = X[bins_before:, :, :]
    Y = y_kf[bins_before-1:, ]

    training_range = [0, 0.85]
    valid_range = [0.85, 1]

    num_examples_kf = X.shape[0]
    # Note that each range has a buffer of 1 bin at the beginning and end
    # This makes it so that the different sets don't include overlapping data
    training_set = np.arange(int(np.round(training_range[0] * num_examples_kf)) + 1,
                             int(np.round(training_range[1] * num_examples_kf)) - 1)
    valid_set = np.arange(int(np.round(valid_range[0] * num_examples_kf)) + 1,
                          int(np.round(valid_range[1] * num_examples_kf)) - 1)

    # Get training data
    X_train = X[training_set, :]
    y_train = Y[training_set, :]
    # Get validation data
    X_valid = X[valid_set, :]
    y_valid = Y[valid_set, :]
    model_lstm = LSTMRegression(units=400, num_epochs=5)
    # Fit model
    model_lstm.fit(X_train, y_train)
    # Get predictions
    y_valid_predicted_lstm = model_lstm.predict(X_valid)
    rho_kf = get_rho(y_valid, y_valid_predicted_lstm)
    print( rho_kf[0:2] ** 2)
    print()
