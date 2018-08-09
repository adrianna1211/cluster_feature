import numpy as np
import os
import pickle
import copy


def load_data(data_dir):
    seq_data = dict()
    for csv_name in os.listdir(data_dir):
        csv_no_ext_name = os.path.splitext(csv_name)[0]
        csv_path = os.path.join(data_dir, csv_name)
        csv_data = np.loadtxt(csv_path, delimiter=',')
        seq_data[csv_no_ext_name] = csv_data
    return seq_data


def sample_seq(mot_seq, st, wlen=150, hop=100):
    mot = []
    ns = 0  # N sequence

    seq_keys = list(mot_seq.keys())
    seq_keys.sort()
    # loop each action sequence, i.e. each file
    for k in seq_keys:
        print("sample_seq: ", k)
        mot_seq_data = mot_seq[k]
        start = st
        end = start + wlen
        while end <= mot_seq_data.shape[0]:
            mot.append(mot_seq_data[start:end, :].flatten())
            ns += 1
            start += hop
            end += hop

    return np.array(mot)


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def normalize_complete_data(data, data_mean, data_std):
    data_out = np.divide((data - data_mean), data_std)
    return data_out


def get_data(data_dir, start, seq_len, hop, norm=True, align=True):
    data_type = os.path.basename(data_dir)
    if not align:
        alldata_dir = data_type + '_' + str(seq_len) + '_' + str(hop) + '_no_align.cpkl'
    else:
        alldata_dir = data_type + '_' + str(seq_len) + '_' + str(hop) + '_align.cpkl'
    print('motion type:', data_type)

    if os.path.exists(alldata_dir):
        f = open(alldata_dir, 'rb')
        data = pickle.load(f)
    else:
        print('loading motion data...')
        motion_seq_dict = load_data(data_dir)
        motion_seq = sample_seq(motion_seq_dict, start, seq_len, hop)
        # alignment 去掉每个seg第一帧的expmap的前6维
        if align:
            motion_seq = motion_seq[:, 6:]
        data = motion_seq

        f = open(alldata_dir, 'wb')
        pickle.dump(data, f)
        f.close()
        print('save data in ', alldata_dir)

    if norm:
        data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(data)
        res = normalize_complete_data(data, data_mean, data_std)
    else:
        res = data
    print('motion seq num:{0}，feature dim{1}'.format(res.shape[0], res.shape[1]))

    return res


def get_all_data(data_dir, start, seq_len, hop, norm=True, align=True):
    if not align:
        alldata_dir = 'all_' + str(seq_len) + '_' + str(hop) + 'no_align.cpkl'
    else:
        alldata_dir = 'all_' + str(seq_len) + '_' + str(hop) + 'align.cpkl'
    if os.path.exists(alldata_dir):
        f = open(alldata_dir, 'rb')
        data = pickle.load(f)
    else:
        print('loading motion data...')
        motion_seq = list()
        for type in os.listdir(data_dir):
            type_path = os.path.join(data_dir, type)
            motion_seq_dict = load_data(type_path)
            motion_seq_type = sample_seq(motion_seq_dict, start, seq_len, hop)
            if len(motion_seq) == 0:
                motion_seq = copy.deepcopy(motion_seq_type)
            else:
                motion_seq = np.vstack([motion_seq, motion_seq_type])

        if align:
            motion_seq = motion_seq[:, 6:]
        data = motion_seq

        f = open(alldata_dir, 'wb')
        pickle.dump(data, f)
        f.close()
        print('save data in ', alldata_dir)

    if norm:
        data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(data)
        res = normalize_complete_data(data, data_mean, data_std)
    else:
        res = data
    print('motion seq num:{0}，feature dim{1}'.format(res.shape[0], res.shape[1]))

    return res
