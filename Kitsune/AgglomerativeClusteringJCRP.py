
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np

distances = [
[0,	0.4598,	0.4124,	1,	0.3292,	0.3309,	0.9978,	0.3886,	0.2174],
[0.4598,	0,	0.2985,	1,	0.1274,	0.1353,	0.9998,	0.2081,	0.1535],
[0.4124,	0.2985,	0,	1,	0.0931,	0.0932,	0.9955,	0.137,	0.1527], 
[1,	1,	1,	0,	1,	1,	1,	1,	1,],
[0.3292,	0.1274,	0.0931,	1,	0,	0.0235,	0.9977,	0.047,	0.3293],
[0.3309,	0.1353,	0.0932,	1,	0.0235,	0,	0.9963,	0.0544,	0.1856],
[0.9978,	0.9998,	0.9955,	1,	0.9977,	0.9963,	0,	0.711,	0.4821],
[0.3886,	0.2081,	0.137,	1,	0.047,	0.0544,	0.711,	0,	0.2566],
[0.2174,	0.1535,	0.1527,	1,	0.3293,	0.1856,	0.4821,	0.2566,	0],
]

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#distances = np.ones(shape = (9,9)) - np.eye(9) - distances
#print(distances)
model = AgglomerativeClustering(affinity='precomputed', n_clusters = 5, linkage = 'complete', compute_distances= True).fit(distances)
print(model.labels_)
print(model.children_)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Device")
plt.gca().axes.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig('./Kitsune/Graphs/AgglomerativeClustering.pdf')

