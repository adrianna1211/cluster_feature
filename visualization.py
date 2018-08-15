from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as cl
import argparse
import os
import socket
import data_utils as util

parse = argparse.ArgumentParser()
parse.add_argument('type', help='dataset type', nargs='?', default='all', type=str)
parse.add_argument('feature', help='dataset feature:exp, kinetic or laban', nargs='?', default='all', type=str)
parse.add_argument('start', help='segment start frame', nargs='?', type=int, default='1')
parse.add_argument('length', help='segment len', nargs='?', type=int, default='30')
parse.add_argument('hop', help='segment hop', nargs='?', type=int, default='30')
parse.add_argument('sample', help='down sample or not', nargs='?', type=int, default='0')
parse.add_argument('k', help='cluster number', nargs='?', type=int, default='160')
args = parse.parse_args()

if socket.gethostname() == 'capg-162':
    root = 'Y:/'
else:
    root = '/mnt/mypublic/'
# data_dir = '/mnt/mypublic/music_dance/dataset/capg/beat_tracking/motion/expmap_15/chaoxianzu'
# data_dir = '/mnt/mypublic/wuxinyue/frame/motion/exp/'
# data_dir = '/mnt/mypublic/wuxinyue/frame/motion/exp_laban_kinetic_frame/chaoxianzu'

data_root = os.path.join(root, 'wuxinyue/frame/motion/exp_laban_kinetic_frame/')
# data_root = '/home/wuxinyue/dataset/cluster/'

# data_root = '/home/wuxinyue/dataset/cluster/'
data_dir = os.path.join(data_root, args.type)

start = args.start
seq_len = args.length
hop = args.hop
sample = args.sample
feat = args.feature

arg_list = [args.type, feat, str(seq_len), str(hop), 'sample' + str(sample), 'k' + str(args.k)]
fig_name = '_'.join(arg_list)

if args.type != 'all':
    X = util.get_data(data_dir, start, seq_len, hop, align=True, feat=feat, sample_rate=sample)
else:
    X, type_label, type_name = util.get_all_data(data_root, start, seq_len, hop, align=True, feat=feat,
                                                 sample_rate=sample,
                                                 type_label=True)

pca_100 = PCA(n_components=100)
pca_result_100 = pca_100.fit_transform(X)
print(
    'Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_100.explained_variance_ratio_)))
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

Y = tsne.fit_transform(pca_result_100)
[n, dim] = X.shape
# range_n_clusters = list(range(5, 10)) + list(range(10, min(n, 100), 5))
#
# if args.type == 'all':
#     plt.rcParams['image.cmap'] = 'nipy_spectral'
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     # plt.figure(figsize=(7, 6))
#     x = ax.scatter(Y[:, 0], Y[:, 1], c=type_label, alpha=0.8, s=25)
#     ccm = x.get_cmap()
#     circles = [Line2D(range(1), range(1), color='w', marker='o', markersize=10, markerfacecolor=item) for item in
#                ccm(np.array(range(len(type_name))) / (len(type_name) - 1))]
#     chartbox = ax.get_position()
#     ax.set_position([chartbox.x0, chartbox.y0, chartbox.width * 0.8, chartbox.height])
#     ax.legend(circles, type_name, loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
#     # plt.show()
#     plt.savefig('./pics/' + fig_name + '_type.png')

colormap = 'nipy_spectral'
# range_n_clusters = [args.k]
# for i, n_clusters in enumerate(range_n_clusters):
#     fig, ax1 = plt.subplots(1, 1)
#     clusterer = KMeans(n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     cm = plt.get_cmap(colormap)
#     cNorm = cl.Normalize(vmin=min(cluster_labels), vmax=max(cluster_labels))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#     ax1.scatter(Y[:, 0], Y[:, 1], c=scalarMap.to_rgba(cluster_labels))

n_clusters = [args.k]
clusterer = KMeans(n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
fig2, ax2 = plt.subplots(1, 1)
y_lower = 10
for j in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == j]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cmx.nipy_spectral(float(j) / n_clusters)
    ax2.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
    # Compute the new y_lower for next plot

# plt.show()
plt.savefig('./pics/' + fig_name + '_distribute.png')
# plt.show()

# np.savetxt(fig_name + '.txt', score)
