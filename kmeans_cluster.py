from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os

import data_utils as util

parse = argparse.ArgumentParser()
parse.add_argument('type', help='dataset type', nargs='?', default='salsa', type=str)
parse.add_argument('feature', help='dataset feature:exp, kinetic or laban', nargs='?', default='all', type=str)
parse.add_argument('start', help='segment start frame', nargs='?', type=int, default='1')
parse.add_argument('length', help='segment len', nargs='?', type=int, default='30')
parse.add_argument('hop', help='segment hop', nargs='?', type=int, default='30')
parse.add_argument('sample', help='down sample or not', nargs='?', type=int, default='0')
parse.add_argument('k', help='cluster number', nargs='?', type=int, default=None)
args = parse.parse_args()

# data_dir = '/mnt/mypublic/music_dance/dataset/capg/beat_tracking/motion/expmap_15/chaoxianzu'
# data_dir = '/mnt/mypublic/wuxinyue/frame/motion/exp/'
# data_dir = '/mnt/mypublic/wuxinyue/frame/motion/exp_laban_kinetic_frame/chaoxianzu'

data_root = '/mnt/mypublic/wuxinyue/frame/motion/exp_laban_kinetic_frame/'
# data_root = '/home/wuxinyue/dataset/cluster/'

data_dir = os.path.join(data_root, args.type)

start = args.start
seq_len = args.length
hop = args.hop
sample = args.sample
feat = args.feature

arg_list = [args.type, feat, str(seq_len), str(hop), 'sample' + str(sample)]
fig_name = '_'.join(arg_list)

if args.type != 'all':
    X = util.get_data(data_dir, start, seq_len, hop, align=True, feat=feat, sample_rate=sample)
else:
    X = util.get_all_data(data_root, start, seq_len, hop, align=True, feat=feat, sample_rate=sample)

# X [n_seg, dim]
# X = util.get_all_data(data_dir, start, seq_len, hop, align=True, sample_rate=False)
# X = util.get_data(data_dir, start, seq_len, hop, align=True, feat='laban')
# X = util.get_data(data_dir, start, seq_len, hop, align=True, feat='kinetic')
# X = util.get_data(data_dir, start, seq_len, hop, align=True, feat='exp', sample_rate=sample)
# X = util.get_data(data_dir, start, seq_len, hop, align=True)

# X, y = make_blobs(n_samples=500,
#                   n_features=2,
#                   centers=4,
#                   cluster_std=1,
#                   center_box=(-10.0, 10.0),
#                   shuffle=True,
#                   random_state=1)  # For reproducibility

[n, dim] = X.shape
print('type:{},length:{},num_seg:{}'.format(args.type, args.length, n))
# chaoxianzu
if not args.k:
    # range_n_clusters = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 175, 200]
    # range_n_clusters = [30, 90, 150, 200]
    range_n_clusters = list(range(5, 10)) + list(range(10, min(n, 205), 5))
else:
    range_n_clusters = [args.k]
# range_n_clusters = list(range(10, 500, 5))

score = list()
for n_clusters in range_n_clusters:
    # fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(9, 15)
    # ax1.set_xlim([-0.5, 1])
    # # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # # plots of individual clusters, to demarcate them clearly.
    # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # sample_silhouette_values = silhouette_samples(X, cluster_labels)
    #
    # y_lower = 10
    # for i in range(n_clusters):
    #     # Aggregate the silhouette scores for samples belonging to
    #     # cluster i, and sort them
    #     ith_cluster_silhouette_values = \
    #         sample_silhouette_values[cluster_labels == i]
    #
    #     ith_cluster_silhouette_values.sort()
    #
    #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #     y_upper = y_lower + size_cluster_i
    #
        # color = cm.nipy_spectral(float(i) / n_clusters)
    #     ax1.fill_betweenx(np.arange(y_lower, y_upper),
    #                       0, ith_cluster_silhouette_values,
    #                       facecolor=color, edgecolor=color, alpha=0.7)
    #
    #     # Label the silhouette plots with their cluster numbers at the middle
    #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    #
    #     # Compute the new y_lower for next plot
    #     y_lower = y_upper + 10  # 10 for the 0 samples
    #     # Compute the new y_lower for next plot
    # plt.show()

plt.plot(range_n_clusters, score, marker='o')
for a, b in zip(range_n_clusters, score):
    plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom')

plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.savefig('./pics/'+fig_name + '.png')
# plt.show()

np.savetxt(fig_name + '.txt', score)
