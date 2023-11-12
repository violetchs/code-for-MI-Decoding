import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as optimize

from numpy.linalg import inv as inv
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import keras
keras_v1=int(keras.__version__[0])<=1
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
matplotlib.use('TkAgg')

deg_step = 30

def tuning_regression(xrange, firing_rate):
    def target_func(x, r0, rmax, theta):
        return r0 + (rmax - r0) * np.cos((x  - theta) * np.pi/180)
    p0 = [25, 50, 0]
    para, _ = optimize.curve_fit(target_func, xrange, firing_rate, p0=p0)
    x = [i for i in range(-180, 180)]
    y = [target_func(a, *para) for a in x]
    fr_pred = [target_func(a, *para) for a in xrange]
    return [x,y, xrange, fr_pred]

def velocity_regression(v, firing_rate):

    model = LinearRegression()  # 构建线性模型
    model.fit(v, firing_rate)  # 自变量在前，因变量在后
    R2 = model.score(v, firing_rate)  # 拟合程度 R2
    #print('R2 = %.2f' % R2)  # 输出 R2
    coef = model.coef_  # 斜率
    intercept = model.intercept_
    return coef, intercept, R2

def spike_toarray(nchannel, spikes, Data):
    '''
    :param nchannel: nchannels
    :param spikes: HDF5 dataset，shape(5,96)
    :param Data: HDF5 dataset，
    :return: spike_neuron: array of neuron spike train
    {neuron1：spikes [1, S], ... },
    neuron_num: num of neurons in each channel
    '''
    spike_neuron = {}
    neuron_num = np.zeros(nchannel)
    for c in range(nchannel):
        for n in range(1, spikes.shape[0]):

            spike = np.array(Data[spikes[n, c]])
            if np.size(spike.shape) == 2:
                if spike.shape[-1] > 100:
                    neuron_num[c] = neuron_num[c]+1
                    key = "c{}_n{}".format(int(c+1),int(neuron_num[c]))
                    spike_neuron[key] = spike
    return spike_neuron, neuron_num

def trail_split(target_pos, npoints):
    '''
    :param target_pos: target position (2, times)
    :param npoints: points of times
    :return: splited trails
            {pos_begin: begin pos of the trail
            orientation: orientation of movement
            t_series: begin point of the trail
            }
    '''
    trails = {}
    pos_begin = np.array([target_pos[0, 0], target_pos[1, 0]])
    pos = [pos_begin]
    t_series = []
    orientation = []
    for point in range(npoints):
        if pos_begin[0] != target_pos[0, point] or pos_begin[1] != target_pos[1, point]:

            pos_begin = target_pos[:, point]
            orientation.append(math.atan2(pos_begin[1]-pos[-1][1], pos_begin[0]-pos[-1][0]) / math.pi * 180)
            pos.append(pos_begin)
            t_series.append(point)



    trails['pos_begin'] = pos
    trails['t_begin'] = t_series
    trails['orientation'] = orientation
    return trails

def divide_trail_by_orientation(trails_info, step = deg_step):
    '''

    :param trails_info: splited trails
            {pos_begin: begin pos of the trail
            orientation: orientation of movement
            t_series: begin point of the trail
            }
    :return: divided trail index by orientation
        {degree_range:  degree range of orientation
        ...
        }
    '''

    index_divided = {}
    for deg in range(-180, 180, step):
        key = '{}_{}'.format(deg, deg + step)
        index_divided[key] = []
    for tl in range(len(trails_info['orientation'])):

        for deg in range(-180, 180, step):
            if trails_info['orientation'][tl]>deg and trails_info['orientation'][tl]<=deg+step:
                key = '{}_{}'.format(deg, deg + step)
                index_divided[key].append(tl)
    return index_divided

def divide_spike_by_orientation(trails_info, trail_divided, npoints, t, spike_neuron, neuron_num_perch, step = deg_step):
    '''

    :param trails_info: splited trails
            {pos_begin: begin pos of the trail
            orientation: orientation of movement
            t_series: begin point of the trail
            }
    :param trail_divided: divided trail index by orientation
        {degree_range:  degree range of orientation
        ...
        }
    :param npoints: points of time sequence
    :param t: times
    :param spike_neuron: array of neuron spike train
    {neuron1：spikes [1, S], ... }
    :param neuron_num_perch: neuron num in each channel
    :param step: degree step for dividing
    :return: spikes divided by different movement orientation
    {degree_range: [(one_trail)[    (one_neuron)[spikes ], ...]]
    ...
    }
    '''
    ntrails = len(trails_info['t_begin'])
    spike_devided = {}
    t_devided = {}
    nchannel = neuron_num_perch.shape[0]
    for deg in range(-180, 180, step):
        key1 = '{}_{}'.format(deg, deg + step)
        trails = trail_divided[key1]
        spike_alltrail_perorien = []
        t_alltrail_perorien = []
        for rl in trails:
            begin_point = trails_info['t_begin'][rl]
            if rl == ntrails-1:
                end_point = npoints - 1
            else:
                end_point = trails_info['t_begin'][rl+1]
            begin_t = t[begin_point]
            end_t = t[end_point]
            spike_onetrail_allneuron = []

            for ch in range(nchannel):
                for n in range(int(neuron_num_perch[ch])):
                    spike_onetrail_perneuron = []
                    key2 = "c{}_n{}".format(ch+1,n+1)
                    spike = spike_neuron[key2]
                    spike = np.squeeze(spike, axis=0)
                    for s in range(spike.shape[0]):
                        if spike[s]>begin_t and spike[s]<end_t:
                            spike_onetrail_perneuron.append(spike[s]-begin_t)
                    spike_onetrail_allneuron.append(spike_onetrail_perneuron)
            spike_alltrail_perorien.append(spike_onetrail_allneuron)
            t_alltrail_perorien.append(end_t-begin_t)
        spike_devided[key1] = spike_alltrail_perorien
        t_devided[key1] = t_alltrail_perorien
    return spike_devided, t_devided

def calcu_kinematic(cursor_pos, p_begin, p_end, kinematic, bin):
    K = []
    if   'pos' in kinematic:
        pos = cursor_pos[:, p_begin: p_end]
        px = np.mean(pos[0, :])
        py = np.mean(pos[1, :])
        K.append(px)
        K.append(py)

    if 'vel' in kinematic:
        pos_begin = cursor_pos[:, p_begin]
        pos_end = cursor_pos[:, p_end - 1]
        vx = (pos_end[0] - pos_begin[0]) / bin
        vy = (pos_end[1] - pos_begin[1]) / bin
        K.append(vx)
        K.append(vy)

    if  'acc' in  kinematic:
        p_mid = int((p_begin+p_end)/2)
        pos_begin1 = cursor_pos[:, p_begin]
        pos_end1 = cursor_pos[:, p_mid]
        pos_begin2 = cursor_pos[:, p_mid]
        pos_end2 = cursor_pos[:, p_end - 1]
        vx1 = (pos_end1[0] - pos_begin1[0]) / bin
        vy1 = (pos_end1[1] - pos_begin1[1]) / bin
        vx2 = (pos_end2[0] - pos_begin2[0]) / bin
        vy2 = (pos_end2[1] - pos_begin2[1]) / bin
        ax = (vx2 - vx1) / bin
        ay = (vy2 - vy1) / bin
        K.append(ax)
        K.append(ay)

    return K

def calcu_finger_kinematic(finger_pos, p_begin, p_end, kinematic, bin):
    K = []
    if   'pos' in kinematic:
        pos = finger_pos[:, p_begin: p_end]
        px = np.mean(-pos[1, :])
        py = np.mean(-pos[2, :])
        K.append(px)
        K.append(py)

    if 'vel' in kinematic:
        pos_begin = finger_pos[:, p_begin]
        pos_end = finger_pos[:, p_end - 1]
        vx = (-pos_end[1] + pos_begin[1]) / bin
        vy = (-pos_end[2] + pos_begin[2]) / bin
        K.append(vx)
        K.append(vy)

    if  'acc' in  kinematic:
        p_mid = int((p_begin+p_end)/2)
        pos_begin1 = finger_pos[:, p_begin]
        pos_end1 = finger_pos[:, p_mid]
        pos_begin2 = finger_pos[:, p_mid]
        pos_end2 = finger_pos[:, p_end - 1]
        vx1 = (-pos_end1[1] + pos_begin1[1]) / bin
        vy1 = (-pos_end1[2] + pos_begin1[2]) / bin
        vx2 = (-pos_end2[1] + pos_begin2[1]) / bin
        vy2 = (-pos_end2[2] + pos_begin2[2]) / bin
        ax = (vx2 - vx1) / bin
        ay = (vy2 - vy1) / bin
        K.append(ax)
        K.append(ay)

    return K

def divide_spike_by_kinematics(cursor_pos, bin, overlap, npoints, t, spike_neuron, neuron_num_perch, kinematic):
    '''

    :param cursor_pos: position of cursor, (2, points) (x, y)
    :param bin: length of time bin
    :param overlap: length of overlap between the time bin
    :param npoints: num of points
    :param t: time series
    :param spike_neuron:
    :param neuron_num_perch: num of neurons in each channel
    :param kinematic: kinematic parameter
    :return:
    '''
    binsize = int(bin * 250)
    step = int(binsize * (1 - overlap))
    spike_samples = []
    kinematics_samples = []
    nchannel = neuron_num_perch.shape[0]
    for ch in range(nchannel):
        for n in range(int(neuron_num_perch[ch])):
            spike_perneuron = []
            key2 = "c{}_n{}".format(ch + 1, n + 1)
            spike = spike_neuron[key2]
            spike = np.squeeze(spike, axis=0)
            s_begin = 0
            for p in range(0, npoints, step):
                spike_trail = []
                if p + step >= npoints:
                    break
                t_begin = t[p]
                t_end = t[p + step]

                for s in range(s_begin, spike.shape[0]):
                    if spike[s] > t_begin and spike[s] < t_end:
                        spike_trail.append(spike[s])
                    if spike[s] >= t_end:
                        s_begin = s
                        break
                fs = len(spike_trail) / bin
                spike_perneuron.append(fs)
            spike_samples.append(spike_perneuron)

    for p in range(0, npoints, step):
        if p + step >= npoints:
            break
        #k = calcu_kinematic(cursor_pos, p, p+step, kinematic, bin)
        k = calcu_finger_kinematic(cursor_pos, p, p+step, kinematic, bin)
        kinematics_samples.append(k)
    return kinematics_samples, spike_samples

def plot_raster(spike_devided, t_devided, if_selected = True,step = deg_step):
    '''

    :param spike_devided: divided by different movement orientation
    {degree_range: [(one_trail)[    (one_neuron)[spikes ], ...]]
    ...
    }
    :param t_devided: durations of each trail
    :param if_selected: if if_selected == True, plot rasterplots with selected neurons
    , otherwise with all neurons.
    :param step: degree step for dividing
    '''
    neuron_list = [0, 1, 2, 18, 24, 25, 27]
    for deg in range(-180, 180, step):
        key = '{}_{}'.format(deg, deg + step)
        spike_train = spike_devided[key]
        t_train = t_devided[key]
        ntrails = len(spike_train)
        nneurons = len(spike_train[0])
        lineoffset = [0]
        linelengths = [0.2]
        if not if_selected == True:
            neuron_list = [x for x in range(nneurons)]
        for n in range(nneurons):
            if n in neuron_list:
                lineoffset[0] = lineoffset[0] + 0.3
                spike = []
                '''t_add = - t_train[0] - 1
                for tl in range(ntrails):
                    t_add = t_add + t_train[tl] + 1
                    if len(spike_train[tl][n]) > 0:
                        for s in spike_train[tl][n]:
                            spike.append(s + t_add)'''
                tl = 0
                for s in spike_train[tl][n]:
                    spike.append(s)
                if len(spike) > 0:
                    data = np.array(spike)

                    plt.eventplot(data, colors='black', lineoffsets=lineoffset, linelengths=linelengths, linewidths=5)
        plt.axvline(x=0, ymin=0, ymax=lineoffset[0]+0.2, linestyle='--', linewidth=5, color='black')
        plt.tick_params(labelsize=11)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        path = './result/raster/{}'.format(deg_step)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, '{}.png').format(key), dpi=600)
        #plt.show()
        plt.close()

def plot_psth(spike_devided, step=deg_step):
    '''

    :param spike_devided:  divided by different movement orientation
    {degree_range: [(one_trail)[    (one_neuron)[spikes ], ...]]
    ...
    }
    :param step: degree step for dividing
    '''
    for deg in range(-180, 180, step):
        key = '{}_{}'.format(deg, deg + step)
        spike_train = spike_devided[key]
        start_time = 0
        duration = 1
        ntrails = len(spike_train)
        nneurons = len(spike_train[0])
        bin_size = 0.1
        num_bins = int(duration / bin_size)
        for n in range(nneurons):
            average_counts = []
            for tl in range(ntrails):
                spike_trail = np.array(spike_train[tl][n])
                spike_trail = spike_trail[spike_trail<=duration]
                trail_counts, _ = np.histogram(spike_trail, bins=num_bins, range=(start_time, start_time + duration))
                average_counts.append(trail_counts)
            average_counts = (sum(average_counts) / len(average_counts))
            bin_centers = np.linspace(start_time + bin_size / 2, start_time + duration - bin_size / 2, num_bins)
            plt.bar(bin_centers, average_counts, width=bin_size)
            plt.xlabel('Time (s)')
            plt.ylabel('Average Spike Count')
            plt.ylim(0, 3)
            plt.title('Peri-Stimulus Time Histogram')
            path = './result/psth/n{}'.format(n)
            if not os.path.exists(path):
                os.mkdir(path)
            fig_path = os.path.join(path, 'key{}'.format(key))
            plt.savefig(fig_path, dpi=600)
            plt.close()

def plot_tuningcurve(spike_devided, t_devided, step=deg_step):
    '''

    :param spike_devided:
    divided by different movement orientation
    {degree_range: [(one_trail)[    (one_neuron)[spikes ], ...]]
    ...
    }
    :param t_devided: durations of each trail
    :param step: degree step for dividing
    '''
    spike_train = spike_devided['0_{}'.format(deg_step)]
    nneurons = len(spike_train[0])
    xrange = [deg for deg in range(-180, 180, step)]
    R2 = []
    for n in range(nneurons):
        firing_rate = []
        for deg in range(-180, 180, step):
            key = '{}_{}'.format(deg, deg + step)
            spike_train = spike_devided[key]
            ntrails = len(spike_train)
            t_train = t_devided[key]
            firing_rate_trails = []
            for tl in range(ntrails):
                spike_trail = spike_train[tl][n]
                duration = t_train[tl]
                fs = len(spike_trail)/duration
                firing_rate_trails.append(fs)
            trail_average_rate = np.mean(np.array(firing_rate_trails))
            firing_rate.append(trail_average_rate)
        plt.plot(xrange, firing_rate, 'o', color = 'black')
        x,y, xrange, fr_pred =tuning_regression(xrange, firing_rate)
        r2 = r2_score(firing_rate, fr_pred)
        R2.append(r2)
        print("{}: {}".format(n, r2))
        plt.plot(x, y, color='black')
        plt.xlabel('Direction of Movement/degree')
        plt.ylabel('Firing rate')
        path = './result/tuning_curve/{}'.format(deg_step)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, 'n{}'.format(n)), dpi=600)
        plt.close()
    R2 = np.array(R2)
    sum1 = R2[np.where(R2>=0.5)].shape[0]
    sum2 = R2[np.where(R2 < 0.5)].shape[0]
    print('r2>=0.5 : {}, r2<0.5 : {}'.format(sum1, sum2))

def neuron_encoding_cursor(cursor_pos, spike_neuron, neuron_num_perch, t, npoints):
    '''
    :param cursor_pos: position of cursor at each t (2, npoints)(x, y)
    :param spike_neuron: array of neuron spike train
    {neuron1：spikes [1, S], ... }
    :param t: time series
    :param npoints: num of points
    :return:
    '''
    bin = 0.1
    overlap = 0
    kinematics, spike_samples = divide_spike_by_kinematics(cursor_pos, bin,
                                                overlap, npoints, t,
                                                spike_neuron, neuron_num_perch, kinematic='acc')
    nneuron = int(sum(neuron_num_perch))
    #plot_tuning(kinematics, spike_samples, nneuron)
    coeff = np.zeros((nneuron, 3))
    R2 = []

    '''train encoded model for each neuron'''
    for n in range(nneuron):
        firing_rate = np.array(spike_samples[n])
        v = np.array(kinematics)
        coef, intercept, r2 = velocity_regression(v[1:, :], firing_rate[:-1])
        b0 = intercept
        b1 = coef[0]
        b2 = coef[1]
        R2.append(r2)
        if n == 54:
            plot_linear_fit(v[1:, :], b0, b1, b2)
        print('{}_r2:{}'.format(n, r2))

def plot_linear_fit(kinematics, b0, b1, b2):
    kx = kinematics[:, 0]
    ky = kinematics[:, 1]
    kxmin = np.min(kx)
    kymin = np.min(ky)
    kxmax = np.max(kx)
    kymax = np.max(ky)
    kxstep = (kxmax - kxmin) / 500
    kystep = (kymax - kymin) / 500
    kxrange = [round(kxmin + (i + 0.5) * kxstep, 2) for i in range(500)]
    kyrange = [round(kymin + (i + 0.5) * kystep, 2) for i in range(500)]
    spikes = np.zeros((500, 500))
    for i in range(500):
        for j in range(500):
            spikes[i, j] = kxrange[i] * b1 + kyrange[j] * b2 + b0
    path = './result/linear_fit'
    plt.imshow(spikes)
    k_mean = np.mean(kinematics, axis=0)
    k_std = np.std(kinematics, axis=0)
    kinematics = (kinematics - k_mean) / k_std
    kx = kinematics[:, 0]
    ky = kinematics[:, 1]
    kxmin = np.min(kx)
    kymin = np.min(ky)
    kxmax = np.max(kx)
    kymax = np.max(ky)
    kxstep = (kxmax - kxmin) / 500
    kystep = (kymax - kymin) / 500
    kxrange = [round(kxmin + (i + 31/2) * kxstep, 2) for i in range(0, 500, 33)]
    kyrange = [round(kymin + (i + 31/2) * kystep, 2) for i in range(0, 500, 33)]
    plt.xticks([i for i in range(0, 500, 33)], kxrange)
    plt.yticks([i for i in range(0, 500, 33)], kyrange)
    plt.tick_params(labelsize=6)
    plt.xlabel("acc_x")
    plt.ylabel("acc_y")
    plt.colorbar()
    plt.savefig(os.path.join(path, 'acc.png'), dpi=600)
    plt.close()

def plot_tuning(kinematics, spike_samples, nneuron):
    kinematics = np.array(kinematics)
    k_mean = np.mean(kinematics, axis=0)
    k_std = np.std(kinematics, axis=0)
    kinematics = (kinematics - k_mean)/k_std
    spike_samples = np.array(spike_samples)
    kx = kinematics[:, 0]
    ky = kinematics[:, 1]
    kxmin = np.min(kx)
    kymin = np.min(ky)
    kxmax = np.max(kx)
    kymax = np.max(ky)
    spikes  = np.zeros((nneuron, 16, 16))
    num = np.zeros((nneuron, 16, 16)) + 1e-8
    kxstep = (kxmax - kxmin)/16
    kystep = (kymax - kymin)/16
    kxrange = [round(kxmin + (i+0.5) *kxstep, 2) for i in range(16)]
    kyrange = [round(kymin + (i+0.5) *kystep, 2) for i in range(16)]
    for n in range(nneuron):
        for s in range(spike_samples.shape[1]):
            i=0
            j=0
            for i in range(16):
                if kx[s]>kxmin + i *kxstep and kx[s] <=kxmin + (i+1)*kxstep:
                    break
            for j in range(16):
                if ky[s] > kymin + j * kystep and ky[s] <= kymin + (j + 1) * kystep:
                    break
            spikes[n, i, j] = spikes[n, i, j] + spike_samples[n, s]
            num[n, i, j] = num[n, i, j] + 1
    spikes = spikes / num
    path = './result/tuning_plot'
    for n in range(nneuron):

        plt.imshow(spikes[n, :, :])
        plt.xticks([i for i in range(16)], kxrange)
        plt.yticks([i for i in range(16)], kyrange)
        plt.tick_params(labelsize=6)
        plt.xlabel("pos_x")
        plt.ylabel("pos_y")
        plt.colorbar()
        plt.savefig(os.path.join(path, 'pos_{}.png'.format(n)), dpi=600)
        plt.close()


def velocity_encoding_finger(finger_pos, spike_neuron, neuron_num_perch, t, npoints):
    '''
    :param finger_pos: position of cursor at each t (3, npoints)(z, -x, -y)
    or (6, npoints)(z, -x, -y, azimuth, elevation, roll)
    :param spike_neuron: array of neuron spike train
    {neuron1：spikes [1, S], ... }
    :param t: time series
    :param npoints: num of points
    :return:
    '''
    bin = 0.5
    overlap = 0.25
    binsize = int(bin * 250)
    step = int(binsize*(1 - overlap))
    spike_samples = []
    velocity_samples = []
    nchannel = neuron_num_perch.shape[0]
    for ch in range(nchannel):
        for n in range(int(neuron_num_perch[ch])):
            spike_perneuron = []
            key2 = "c{}_n{}".format(ch + 1, n + 1)
            spike = spike_neuron[key2]
            spike = np.squeeze(spike, axis=0)
            s_begin = 0
            for p in range(0, npoints, step):
                spike_trail = []
                if p+step >= npoints:
                    break
                t_begin = t[p]
                t_end = t[p+step]

                for s in range(s_begin, spike.shape[0]):
                    if spike[s] > t_begin and spike[s] < t_end:
                        spike_trail.append(spike[s])
                    if spike[s] >= t_end:
                        s_begin = s
                        break
                fs = len(spike_trail) / bin
                spike_perneuron.append(fs)
            spike_samples.append(spike_perneuron)

    for p in range(0, npoints, step):
        if p + step >= npoints:
            break
        pos_begin = finger_pos[:,p]
        pos_end = finger_pos[:, p+step]
        vx = (-pos_end[1] + pos_begin[1]) / bin
        vy = (-pos_end[2] + pos_begin[2]) / bin

        velocity_samples.append([vx, vy])
    nneuron = int(sum(neuron_num_perch))
    coeff = np.zeros((nneuron, 3))
    for n in range(nneuron):
        firing_rate = np.array(spike_samples[n])
        v = np.array(velocity_samples)
        coef, intercept, R2 = velocity_regression(v, firing_rate)
        b0 = intercept
        b1 = coef[0]
        b2 = coef[1]

def plot_r2(R2_1, R2_2):
    x_range = [round(0.05 * i, 2) for i in range(7)]
    num1 = np.zeros(6)
    num2 = np.zeros(6)
    for r in R2_1:
        for i in range(6):
            if r >= x_range[i] and r < x_range[i+1]:
                num1[i] = num1[i] + 1
    for r in R2_2:
        for i in range(6):
            if r >= x_range[i] and r < x_range[i+1]:
                num2[i] = num2[i] + 1
    bar_width = 0.25  # 条形宽度
    index_r2_1 = np.arange(6)+0.375  # 男生条形图的横坐标
    index_r2_2 = index_r2_1 + bar_width
    plt.bar(index_r2_1, num1, width=bar_width, label='pva model')
    plt.bar(index_r2_2, num2, width=bar_width, label='pv model')
    plt.xticks(np.arange(7), x_range)
    plt.ylabel('neuron num')
    plt.legend()
    plt.show()
    plt.close()


class KalmanFilterRegression(object):

    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self,C1=1,C2=1):
        self.C1=C1
        self.C2=C2

    def fit(self,X_kf_train,y_train):

        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)

        #number of time bins
        nt=X.shape[1]

        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C1 #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        #In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) #Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt/self.C2 #Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        A,W,H,Q=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        #P=np.matrix(np.diag([2] * num_states))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))

        #Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            #Do first part of state update - based on transition matrix
            P_m=A*P*A.T+W
            state_m=A*state

            #Do second part of state update - based on measurement matrix
            K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state) #Record state at the timestep
        y_test_predicted=states.T
        return y_test_predicted

########## R-squared (R2) ##########

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s




########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos

class LSTMRegression(object):

    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted