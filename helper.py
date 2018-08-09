# -*- coding:utf-8 -*-  
import numpy as np
import os
import reader as rd
import laban_feature_frame as lbf
import copy


def get_joint_dict(skel='capg'):
    joint_dict = dict()
    if skel == 'capg':
        joint_dict = {
            'Hips': 'Hips',
            'LowerBack': 'Chest',
            'Spine': 'Chest2',
            'Neck': 'Neck',
            'Head': 'Head',
            'LeftCollar': 'LeftCollar',
            'LeftShoulder': 'LeftShoulder',
            'LeftElbow': 'LeftElbow',
            'LeftHand': 'LeftWrist',
            'RightCollar': 'RightCollar',
            'RightShoulder': 'RightShoulder',
            'RightElbow': 'RightElbow',
            'RightHand': 'RightWrist',
            'LeftHip': 'LeftHip',
            'LeftKnee': 'LeftKnee',
            'LeftFoot': 'LeftAnkle',
            'RightHip': 'RightHip',
            'RightKnee': 'RightKnee',
            'RightFoot': 'RightAnkle',
        }
    elif skel == 'groovenet':
        joint_dict = {
            'Hips': 'Hips',
            'LowerBack': 'LowerBack',
            'Spine': 'Spine',
            'Neck': 'Neck1',
            'Head': 'Head',
            'LeftCollar': 'LeftShoulder',
            'LeftShoulder': 'LeftArm',
            'LeftElbow': 'LeftForeArm',
            'LeftHand': 'LeftHand',
            'RightCollar': 'RightShoulder',
            'RightShoulder': 'RightArm',
            'RightElbow': 'RightForeArm',
            'RightHand': 'RightHand',
            'LeftHip': 'LeftUpLeg',
            'LeftKnee': 'LeftLeg',
            'LeftFoot': 'LeftFoot',
            'RightHip': 'RightUpLeg',
            'RightKnee': 'RightLeg',
            'RightFoot': 'RightFoot',
        }
    return joint_dict


def read_bvh(bvh_path):
    read_inst = rd.MyReader(bvh_path, False)
    points, channels, nodes, node_order, limits = read_inst.read()
    node_dict = dict()
    for i, node in enumerate(nodes):
        node_dict[node.name] = i

    points = np.array(points)

    return points, channels, nodes, node_order, node_dict, limits


def get_lf_obj(points, channels, nodes, node_order, skel='capg'):
    joint_dict = get_joint_dict(skel)
    lf = lbf.LabanFeature(points, channels, nodes, node_order, joint_dict)
    return lf


def get_laban_feature(lf):
    shape_feat = lf.get_shape_feature()
    body_feat = lf.get_body_feature()
    effort_feat = lf.get_effort_feature()
    space_feat = lf.get_space_feature()

    laban_feat = []
    laban_feat.extend(body_feat)
    laban_feat.extend(space_feat)
    laban_feat.extend(shape_feat)
    laban_feat.extend(effort_feat)

    return laban_feat


def load_data(data_dir, files_list):
    data = []
    for file_name in files_list:  # 15
        print("Reading file {0}".format(file_name))
        file_path = data_dir + '/' + file_name
        if not os.path.exists(file_path):
            continue
        sequence = np.loadtxt(file_path)  # 返回float32格式，大约小数点后7位
        print(sequence.shape)
        n, d = sequence.shape
        print("sequence shape:[{},{}]".format(n, d))
        if len(data) == 0:
            data = copy.deepcopy(sequence)
        else:
            data = np.append(data, sequence, axis=0)

    return data


def load_frame_data(data_dir, files_list):
    data = []
    for file_name in files_list:  # 15
        print("Reading file {0}".format(file_name))
        file_path = data_dir + '/' + file_name
        if not os.path.exists(file_path):
            continue
        sequence = np.loadtxt(file_path)  # 返回float32格式，大约小数点后7位
        # print(sequence.shape)
        n, d = sequence.shape
        # print("sequence shape:[{},{}]".format(n, d))
        if len(data) == 0:
            data = copy.deepcopy(sequence)
        else:
            data = np.vstack([data, sequence])

    return data


def load_test_data(gt_dir, gen_dir, files_list):
    data = []
    for file_name in files_list:
        print("Reading file {0}".format(file_name))
        gt_path = gt_dir + '/' + file_name
        gen_path = gen_dir + '/' + file_name
        if not os.path.exists(gen_path):
            continue
        gt_seq = np.loadtxt(gt_path)
        gen_seq = np.loadtxt(gen_path)
        n_seg = min(gt_seq.shape[0], gen_seq.shape[0])
        gt_seq = gt_seq[:n_seg, :]
        n, d = gt_seq.shape
        print("sequence shape:[{},{}]".format(n, d))
        if len(data) == 0:
            data = copy.deepcopy(gt_seq)
        else:
            data = np.append(data, gt_seq, axis=0)

    return data

def normalization_stats_maxmin(complete_data):
    data_max = np.max(complete_data, axis=0)
    data_min = np.min(complete_data, axis=0)
    return data_max, data_min,

def normalization_stats(complete_data):
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    data_max = np.max(complete_data, axis=0)
    data_min = np.min(complete_data, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0
    return data_mean, data_std, data_max, data_min, dimensions_to_ignore, dimensions_to_use

def normalize_data(data, data_mean, data_std, dim2use, reduce_dim=False):
    data_out = np.divide((data - data_mean), data_std + 1e-12)
    if not reduce_dim:
        return data_out
    else:
        return data_out[:, dim2use]


def read_data(train_files, test_files, gt_dir, gen_dir):
    """
    读取train file ，得到均值方差，并返回normalize之后的test file
    :param train_files:
    :param test_files:
    :param gt_dir: ground-truth directory
    :param gen_dir: generation directory
    :return:
    """
    train_set = load_data(gt_dir, train_files)
    test_set = load_test_data(gt_dir, gen_dir, test_files)
    data_mean, data_std, dim2ignore, dim2use = normalization_stats(train_set)
    # train_norm = normalize_data(train_set,data_mean,data_std)

    test_norm = normalize_data(test_set, data_mean, data_std, dim2use, True)
    return test_norm, data_mean, data_std, dim2use


def compare_laban(gen, gt, metric):
    if metric == 'mse':
        dist = np.mean(np.linalg.norm(gen - gt, axis=1))
    elif metric == 'cos':  # 1-u*v/|u||v|
        norm_gen = np.linalg.norm(gen, axis=1)
        norm_gt = np.linalg.norm(gt, axis=1)
        dist = 1 - np.mean((gen * gt).sum(axis=1) / (norm_gen * norm_gt))
    return dist


def normalize_max_min(data, data_max, data_min, dim2use, reduce_dim=False):
    data_out = np.divide((data - data_min), (data_max - data_min + 1e-12))
    if not reduce_dim:
        return data_out
    else:
        return data_out[:, dim2use]


def normalize_data_maxmin(data, data_max, data_min):
    data_out = np.devide(data - data_min, data_max - data_min + 1e-12)
    return data_out


def get_laban_feature_frame(lf):
    shape_feat = lf.get_shape_feature_frame(hands_mean=False)
    body_feat = lf.get_body_feature_frame()
    effort_feat = lf.get_effort_feature_frame(lr_mean=False)
    space_feat = lf.get_space_feature_frame()

    laban_feat = body_feat
    laban_feat = np.hstack([laban_feat, space_feat])
    laban_feat = np.hstack([laban_feat, shape_feat])
    laban_feat = np.hstack([laban_feat, effort_feat])

    return laban_feat


def compare_laban(gen, gt, metric):
    if metric == 'mse':
        dist = np.mean(np.linalg.norm(gen - gt, axis=1))
    elif metric == 'cos':  # 1-u*v/|u||v|
        norm_gen = np.linalg.norm(gen, axis=1)
        norm_gt = np.linalg.norm(gt, axis=1)
        dist = 1 - np.mean((gen * gt).sum(axis=1) / (norm_gen * norm_gt))
    return dist
