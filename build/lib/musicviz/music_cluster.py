from pathlib import Path
import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from .feature import Feature_extract


class cluster:

    def __init__(self, datapath, fname = "./features.csv"):
        self.features = None
        self.standard_embedding = None
        self.clusterable_embedding = None
        self.GMM = None
        self.sorted_map = None

        if fname is not None:
            csv_file = Path(fname)
            if csv_file.is_file():
                self.features = np.loadtxt(open(csv_file, "rb"), delimiter=",")
        else:
            fit = Feature_extract()
            self.features = fit.process(datapath, percent=0.1)
            np.savetxt(fname, self.features, delimiter=',')

    def fit(self, graph = False):
        self.standard_embedding = umap.UMAP(n_neighbors=5, random_state=42).fit_transform(self.features)

        self.dim_reduce = umap.UMAP(
            n_neighbors=10,
            min_dist=0.0,
            n_components=5,
            random_state=42,
        )
        self.clusterable_embedding = self.dim_reduce.fit_transform(self.features)

        # standard_embedding = umap.UMAP(random_state=42).fit_transform(features)

        self.GMM = GaussianMixture(n_components=5, random_state=716, n_init=10)
        preds = self.GMM.fit_predict(self.clusterable_embedding)
        # kmeans_labels = cluster.KMeans(n_clusters=5).fit_predict(features)
        if graph is True:
            plt.scatter(self.standard_embedding[:, 0], self.standard_embedding[:, 1], c=preds, s=0.1, cmap='Spectral')
            plt.show()

        print("_G: ", silhouette_score(self.clusterable_embedding, preds))

    def cluster_map(self, base_fd):
        freq = []

        for i in range(5):
            c_n = base_fd + str(i + 1)

            test = Feature_extract()

            c_n_F = test.process(c_n, 5, 0.1)

            c_n_emb = self.dim_reduce.transform(c_n_F)

            c_n_P = self.GMM.predict(c_n_emb)

            freq.append(c_n_P)

        norm_freq = np.zeros([5, 5])

        for i in range(5):
            length = freq[i].shape[0]
            for j in range(5):
                norm_freq[i][j] = sum(freq[i] == j) / length

        normal_const = np.sum(norm_freq, axis=0)
        cluster_map = []
        # print(normal_const)
        order = np.argsort(normal_const)
        for i in order:
            max_index_col = np.argmax(norm_freq[:, i])

            cluster_map.append(max_index_col)
            norm_freq[max_index_col, :] = -1

        print(norm_freq)

        self.sorted_map = np.take(cluster_map, np.argsort(order))