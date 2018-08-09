from numpy import linalg as la
import numpy as np
import math
import copy
# import warnings


class Position:
    def __int__(self, rotation, position):
        self.rotation = rotation
        self.position = position


def three_axis_rot(r11, r12, r21, r31, r32, r11a, r22a, lim='default'):

    r1 = np.arctan2(r11, r12)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #         r2 = np.arcsin(r21)
    #     except Warning:
    #         print(r21)
    r2 = np.arcsin(np.clip(r21, -1., 1.))
    r3 = np.arctan2(r31, r32)

    return r1, r2, r3


def dcm2angle(dcm, order='zxy'):
    order = order.lower()
    if order == 'zxy':
        r1, r2, r3 = three_axis_rot(-dcm[1, 0], dcm[1, 1], dcm[1, 2],
                                    -dcm[0, 2], dcm[2, 2],
                                    dcm[0, 1], dcm[0, 0])
    else:
        raise Exception("Unknown order!")

    return r1, r2, r3


def exp2rot(e):
    theta = la.norm(e)
    eps = 1e-32
    r0 = np.divide(e, theta + eps)
    r0x = np.array([[0, -r0[2], r0[1]],
                    [0, 0, -r0[0]],
                    [0, 0, 0]])
    r0x = r0x - r0x.transpose()
    rot = np.eye(3, 3) + \
        np.sin(theta) * r0x + \
        np.dot((1 - np.cos(theta)) * r0x, r0x)

    return rot


def quat2exp(q):
    eps = 1e-32
    if np.abs(la.norm(q) - 1) > 1e-3:
        raise Exception('quat2exp: input quaternion is not norm 1.')

    sin_half_theta = la.norm(q[1:])
    cos_half_theta = q[0]
    r0 = q[1:] / (la.norm(q[1:]) + eps)
    theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
    theta = np.mod(theta + 2 * np.pi, np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rot2exp(rot):
    return quat2exp(rot2quat(rot))


def rot2quat(r):
    eps = 1e-32
    d = r - r.transpose()
    r_ = np.array([-d[1, 2],
                   d[0][2],
                   -d[0][1]])
    sin_theta = la.norm(r_) / 2
    r0 = r_ / (la.norm(r_) + eps)
    cos_theta = (np.trace(r) - 1) / 2

    theta = np.arctan2(sin_theta, cos_theta)
    q = np.append(np.cos(theta / 2), r0 * np.sin(theta/2))

    return q


def revert_coordinate_space(chls, init_r, init_t):
    rec_chls = copy.deepcopy(chls)
    r_prev = init_r
    t_prev = init_t
    root_rot_ind = [3, 4, 5]

    for i in range(chls.shape[0]):
        r_diff = exp2rot(chls[i][root_rot_ind])
        r = np.dot(r_diff, r_prev)
        rec_chls[i][root_rot_ind] = rot2exp(r)
        t = t_prev + (np.dot(la.inv(r_prev), chls[i, 0:3]).transpose()).transpose()
        rec_chls[i, 0:3] = t

        t_prev = t
        r_prev = r

    return rec_chls


def rad2deg(rad):
    deg = np.multiply(180 / np.pi, rad)
    return deg


def deg2rad(deg):
    rad = deg / 180 * np.pi
    return rad


def exp2euler(skel, exp_chls):
    euler = np.array([])
    for i in range(len(skel)):
        if len(skel[i].children) == 0:
            continue

        if len(skel[i].pos_idx):
            euler = np.append(euler, exp_chls[skel[i].pos_idx])

        exp = exp_chls[skel[i].exp_idx]
        rot = exp2rot(exp)
        [e1, e2, e3] = dcm2angle(rot, skel[i].order)
        eul = [e1, e2, e3]
        eul = rad2deg(eul)
        euler = np.append(euler, eul)
    return euler


def exp2bvh(skel, exp_chls, init_r=np.eye(3, 3), init_t=np.zeros([1, 3])):
    rec_chls = revert_coordinate_space(exp_chls, init_r, init_t)
    bvh_chls = np.array([[]])

    for i in range(rec_chls.shape[0]):
        frame_bvh_chl = exp2euler(skel, rec_chls[i, :])
        if i == 0:
            bvh_chls = np.atleast_2d(frame_bvh_chl)
        else:
            bvh_chls = np.vstack((bvh_chls, frame_bvh_chl))

    return bvh_chls


def rotation_matrix(x_angle, y_angle, z_angle, order='zxy'):
    order = order.lower()
    c1 = math.cos(x_angle)
    c2 = math.cos(y_angle)
    c3 = math.cos(z_angle)
    s1 = math.sin(x_angle)
    s2 = math.sin(y_angle)
    s3 = math.sin(z_angle)
    rm = np.array([[c2*c3-s1*s2*s3, c2*s3+s1*s2*c3, -s2*c1],
                   [-c1*s3, c1*c3, s1],
                   [s2*c3+c2*s1*s3, s2*s3-c2*s1*c3, c2*c1]])
    return rm


def bvh2xyz_frame(skel, channels):
    xyz_struct = []
    for i in range(len(skel)):

        if len(skel[i].pos_idx) > 0:
            x_pos = channels[skel[i].pos_idx[0]]
            y_pos = channels[skel[i].pos_idx[1]]
            z_pos = channels[skel[i].pos_idx[2]]
        else:
            x_pos = 0
            y_pos = 0
            z_pos = 0

        if len(skel[i].rot_idx) > 0:
            x_angle = deg2rad(channels[skel[i].rot_idx[0]])
            y_angle = deg2rad(channels[skel[i].rot_idx[1]])
            z_angle = deg2rad(channels[skel[i].rot_idx[2]])
        else:
            x_angle = 0.
            y_angle = 0.
            z_angle = 0.

        this_rot = rotation_matrix(x_angle, y_angle, z_angle, skel[i].order)
        this_pos = np.array([x_pos, y_pos, z_pos])
        struct = Position()

        xyz_struct.append(struct)

        if i == 0:
            xyz_struct[i].rotation = this_rot
            xyz_struct[i].position = this_pos  # + np.array(skel[i].offset)
        else:
            parent = skel[i].parent
            this_pos = this_pos + np.array(skel[i].offset)
            xyz_struct[i].position = np.dot(this_pos, xyz_struct[parent].rotation) + xyz_struct[parent].position
            xyz_struct[i].rotation = np.dot(this_rot, xyz_struct[parent].rotation)

    points = np.array([])
    for m in xyz_struct:
        points = np.append(points, m.position)

    return points


def bvh2xyz(skel, bvh_chls):
    xyz_chls = np.array([])
    for i in range(bvh_chls.shape[0]):
        xyz_frame_chl = bvh2xyz_frame(skel, bvh_chls[i])
        if i == 0:
            xyz_chls = np.atleast_2d(xyz_frame_chl)
        else:
            xyz_chls = np.vstack((xyz_chls, xyz_frame_chl))

    return xyz_chls


def angle2quat(r1, r2, r3, order='zxy'):
    angles = np.array([r1, r2, r3])
    cang = np.cos(angles/2)
    sang = np.sin(angles/2)
    q = np.array([cang[0] * cang[1] * cang[2] - sang[0] * sang[1] * sang[2],
                 cang[0] * sang[1] * cang[2] - sang[0] * cang[1] * sang[2],
                 cang[0] * cang[1] * sang[2] + sang[0] * sang[1] * cang[2],
                 cang[0] * sang[1] * sang[2] + sang[0] * cang[1] * cang[2],
                  ])

    return q


def bvh2quat(bvh_chls):
    bvh_len = bvh_chls.shape[0]
    quat_chls = np.array([[]])
    for r in range(bvh_len):
        bvh_line = bvh_chls[r, :]
        quat_line = bvh_line[0:3]
        for c in range(3, 58, 3):
            zr = deg2rad(bvh_line[c])
            xr = deg2rad(bvh_line[c+1])
            yr = deg2rad(bvh_line[c+2])
            q = angle2quat(zr, xr, yr)
            quat_line = np.append(quat_line, q)
        if r == 0:
            quat_chls = np.atleast_2d(quat_line)
        else:
            quat_chls = np.vstack((quat_chls, quat_line))

    return quat_chls


def exp2quat(skel, exp_chls):
    bvh_chls = exp2bvh(skel, exp_chls)
    quat_chls = bvh2quat(bvh_chls)

    return quat_chls


def exp2xyz_frame(skel, channels):
    xyz_struct = []
    for i in range(len(skel)):

        if len(skel[i].pos_idx) > 0:
            x_pos = channels[skel[i].pos_idx[0]]
            y_pos = channels[skel[i].pos_idx[1]]
            z_pos = channels[skel[i].pos_idx[2]]
        else:
            x_pos = 0
            y_pos = 0
            z_pos = 0

        if len(skel[i].rot_idx) > 0:
            this_rot = exp2rot(channels[skel[i].exp_idx])
            eul = dcm2angle(this_rot, skel[i].order)
            eul = np.array(eul)
            order = skel[i].order.lower()
            rot_order = [order.find(c) for c in 'xyz']
            [x_angle, y_angle, z_angle] = eul[rot_order]
        else:
            x_angle = 0.
            y_angle = 0.
            z_angle = 0.
        this_rot = rotation_matrix(x_angle, y_angle, z_angle, skel[i].order)
        this_pos = np.array([x_pos, y_pos, z_pos])
        struct = Position()

        xyz_struct.append(struct)

        if i == 0:
            xyz_struct[i].rotation = this_rot
            xyz_struct[i].position = this_pos  # + np.array(skel[i].offset)
        else:
            parent = skel[i].parent
            this_pos = this_pos + np.array(skel[i].offset)
            xyz_struct[i].position = np.dot(this_pos, xyz_struct[parent].rotation) + xyz_struct[parent].position
            xyz_struct[i].rotation = np.dot(this_rot, xyz_struct[parent].rotation)

    points = np.array([])
    for m in xyz_struct:
        points = np.append(points, m.position)

    return points


def exp2xyz(skel, exp_chls):
    init_r = np.eye(3, 3)
    init_t = np.zeros([1, 3])
    rec_chls = revert_coordinate_space(exp_chls, init_r, init_t)

    xyz_chls = np.array([[]])
    for i in range(exp_chls.shape[0]):
        xyz_frame_chl = exp2xyz_frame(skel, rec_chls[i])
        if i == 0:
            xyz_chls = np.atleast_2d(xyz_frame_chl)
        else:
            xyz_chls = np.vstack((xyz_chls, xyz_frame_chl))

    return xyz_chls


def exp2xyz_2(skel, exp_chls):
    bvh_chls = exp2bvh(skel, exp_chls)
    xyz_chls = np.array([[]])
    for i in range(bvh_chls.shape[0]):
        xyz_frame_chl = bvh2xyz_frame(skel, bvh_chls[i])
        if i == 0:
            xyz_chls = np.atleast_2d(xyz_frame_chl)
        else:
            xyz_chls = np.vstack((xyz_chls, xyz_frame_chl))

    return xyz_chls


def xyz2vel(xyz_chls):
    """
    :param xyz_chls: xyz channels: nframes * ndim
    :return: velocity channels:  nframes - 1 * ndim
    """
    # pre_chls: [0, 1, 2, end-2, :]
    # cur_chls: [1, 2, end-1, :]
    if np.atleast_2d(xyz_chls).shape[0] < 2:
        raise Exception("input channels shape0 should greater than 1!")
    pre_chls = xyz_chls[:-1, :]
    cur_chls = xyz_chls[1:, :]
    vel_chls = cur_chls - pre_chls

    return vel_chls


def vel2acc(vel_chls):
    return xyz2vel(vel_chls)


def vel2energy(vel_chls):
    """
    :param vel_chls:
    :return: nframes * (ndim / 3)
    """
    vel_energys = np.array([[]])
    for i in range(vel_chls.shape[0]):
        vel_energy = np.array([])
        for j in range(0, vel_chls.shape[1], 3):
            vel_energy = np.append(vel_energy, la.norm(vel_chls[i, j:j+3]))
        if i == 0:
            vel_energys = np.atleast_2d(vel_energy)
        else:
            vel_energys = np.vstack((vel_energys, vel_energy))

    return vel_energys


def xyz2energy(xyz_chls):
    vel_chls = xyz2vel(xyz_chls)
    vel_energys = vel2energy(vel_chls)

    return vel_energys


def exp2vel(skel, exp_chls):
    xyz_chls = exp2xyz(skel, exp_chls)
    vel_chls = xyz2vel(xyz_chls)
    return vel_chls


def exp2energy(skel, exp_chls):
    xyz_chls = exp2xyz(skel, exp_chls)
    return xyz2energy(xyz_chls)


def xyz2acc(xyz_chls):
    vel_chls = xyz2vel(xyz_chls)
    acc_chls = vel2acc(vel_chls)

    return acc_chls


def exp2ang_vel(exp_chls):
    init_r = np.eye(3, 3)
    init_t = np.zeros([1, 3])
    rec_chls = revert_coordinate_space(exp_chls, init_r, init_t)
    ang_vel_chls = np.array([[]])
    for i in range(1, rec_chls.shape[0]):
        frame_chl = np.array([])
        for j in range(3, rec_chls.shape[1], 3):
            cur_r = exp2rot(rec_chls[i, j:j+3])
            pre_r = exp2rot(rec_chls[i-1, j:j+3])
            r_diff = np.dot(cur_r, la.inv(pre_r))
            exp_diff = rot2exp(r_diff)
            frame_chl = np.append(frame_chl, exp_diff)
        if i == 1:
            ang_vel_chls = np.atleast_2d(frame_chl)
        else:
            ang_vel_chls = np.vstack((ang_vel_chls, frame_chl))
    return ang_vel_chls


def ang_vel2acc(ang_vel_chls):
    # TODO: make sure this implementation correct
    # return xyz2vel(ang_vel_chls)
    return exp2ang_vel(ang_vel_chls)


def exp2ang_acc(exp_chls):
    ang_vel_chls = exp2ang_vel(exp_chls)
    ang_acc_chls = ang_vel2acc(ang_vel_chls)

    return ang_acc_chls
