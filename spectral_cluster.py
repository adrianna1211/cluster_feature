from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import data_utils as util

# data_dir = 'D:\dataset\capg\\frame\motion\strip_exp\chaoxianzu'
data_dir = '/mnt/mypublic/wuxinyue/frame/motion/exp/chaoxianzu'
start = 1
seq_len = 30
hop = 30
# X [n_seg, dim]
X = util.get_data(data_dir, start, seq_len, hop, norm=False)
#
# X, y = make_blobs(n_samples=500,
#                   n_features=2,
#                   centers=4,
#                   cluster_std=1,
#                   center_box=(-10.0, 10.0),
#                   shuffle=True,
#                   random_state=1)  # For reproducibility

# distance matrix
distance_matrix = pairwise_distances(X, metric='mahalanobis')
similarity = np.exp(-distance_matrix / distance_matrix.std())
# chaoxianzu
# range_n_clusters = [2, 3, 4, 5, 6]
# range_n_clusters = [15, 30, 45, 60, 75, 90, 105, 120, 135]
range_n_clusters = list(range(10, 205, 5))

score = list()

for n_clusters in range_n_clusters:
    # fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(18, 3)
    # ax1.set_xlim([-0.1, 1])
    # # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # # plots of individual clusters, to demarcate them clearly.
    # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = SpectralClustering(n_clusters, affinity='precomputed', random_state=10)
    clusterer.fit(similarity)
    cluster_labels = clusterer.labels_.astype(np.int)
    #
    # clusterer = SpectralClustering(n_clusters, affinity='nearest_neighbors', random_state=10)
    # clusterer.fit(X)
    # cluster_labels = clusterer.labels_.astype(np.int)
    # affinity_matrix = clusterer.affinity_matrix_.astype(np.float)
    # diff = distance_matrix - affinity_matrix

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
    #     color = cm.nipy_spectral(float(i) / n_clusters)
    #     ax1.fill_betweenx(np.arange(y_lower, y_upper),
    #                       0, ith_cluster_silhouette_values,
    #                       facecolor=color, edgecolor=color, alpha=0.7)
    #
    #     # Label the silhouette plots with their cluster numbers at the middle
    #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    #
    #     # Compute the new y_lower for next plot
    #     y_lower = y_upper + 10  # 10 for the 0 samples
    # plt.show()

plt.plot(range_n_clusters, score, marker='o')
for a, b in zip(range_n_clusters, score):
    plt.text(a, b + 0.001, '%.2f' % b, ha='center', va='bottom')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.show()
