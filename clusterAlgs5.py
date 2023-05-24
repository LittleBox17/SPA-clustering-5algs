import numpy as np
from numpy import int32
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.neighbors import kneighbors_graph
from collections import Counter
from datasets import DatasetParameters
import trsfile
from scipy.fftpack import fft

class Clustering:

    def __init__(self):
        self.directory = None
        self.dataset = None
        self.dataset_parameters = DatasetParameters()
        self.target_params = None
        self.target_binkey = None
        self.dataseg = None

    def set_directory(self, directory):
        self.directory = directory

    def set_dataset(self, target):
        self.dataset = target
        self.dataset_parameters = DatasetParameters()
        self.target_params = self.dataset_parameters.get_trace_set(self.dataset)
        self.target_binkey = self.dataset_parameters.get_binkey_set(self.dataset)

    def set_database_name(self, database_name):
        self.database_name = database_name

    def sca_parameters(self):
        return self.dataset_parameters

    def load_sets(self):
        print(self.directory)
        if self.dataset == 'sm2':
            # self.dataseg = trsfile.open(self.directory + self.dataset + '.trs')[:]
            self.dataseg = np.load(self.directory + self.dataset + '.npy', allow_pickle=True).astype(float)
        else:
            temp = np.load(self.directory + self.dataset + '_trace_cutted.npy', allow_pickle=True).astype(float)
            self.dataseg = temp[0,:,:]
        return self.dataseg

    def fft_feature(self, segs = None, visualization=False):
        if segs == None:
            segs = self.load_sets()
        SegFFT = []
        for i in range(len(segs)):
            seg = segs[i]
            N = len(segs[i])
            SegFFT.append(fft(seg, N))
        SegFFT = np.abs(SegFFT)

        if visualization:
            plt.figure('FFT Feature')
            plt.subplot(221)
            plt.plot(segs[3])
            plt.title('Operation-Double', size=13)
            plt.ylabel('original seg', size=13)
            plt.xlabel('samples')
            plt.subplot(222)
            plt.plot(segs[4])
            plt.title('Operation-Double', size=13)
            plt.xlabel('samples')
            plt.subplot(223)
            plt.plot(SegFFT[3, :N // 2])
            plt.ylabel('FFT Feature', size=13)
            plt.xlabel('frequency/Hz')
            plt.subplot(224)
            plt.plot(SegFFT[4, :N // 2])
            plt.xlabel('frequency/Hz')

        return SegFFT

    def run_cluster5(self, segs = np.array([]), n_iterations=1, pca_dim = 2, visualization=False, vistype = 'cres'):
        if ~segs.any():
            segs = self.load_sets()
        binkey = self.target_binkey
        # PCA降维
        pca_para = PCA(n_components=pca_dim)
        pca_para.fit(segs)
        ldim_points = pca_para.transform(segs)
        # 聚类参数
        dbscan_para = cluster.DBSCAN(eps=self.target_params["eps"], min_samples=self.target_params["min_samples"])
        bandwidth = cluster.estimate_bandwidth(ldim_points, quantile=self.target_params["quantile"])
        ms_para = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, seeds=self.target_params["seeds"])
        connectivity = kneighbors_graph(
            ldim_points, n_neighbors=self.target_params["n_neighbors"], include_self=False
        )
        connectivity = 0.5 * (connectivity + connectivity.T)
        ward_para = cluster.AgglomerativeClustering(
            n_clusters=self.target_params["n_clusters"], linkage="ward", connectivity=connectivity
        )
        kmeans_para = cluster.MiniBatchKMeans(n_clusters=self.target_params["n_clusters"], init=self.target_params["init_center"])
        gmm_para = mixture.GaussianMixture(
            n_components=self.target_params["n_clusters"], covariance_type=self.target_params["type"]
        )

        clustering_algorithms = (
            ("KMeans", kmeans_para),
            ("MeanShift", ms_para),
            ("DBSCAN", dbscan_para),
            ("WARD", ward_para),
            ("GM", gmm_para),
        )

        results = []
        for n in range(n_iterations):
            print('-------- iter = '+ str(n) +' -------- ')
            iteration_results = {'labels':[], 'errors':[], 'count_T':[], 'count_F':[]}
            if visualization:
                plt.figure()
                plot_num = 1

            for name, algorithm in clustering_algorithms:
                algorithm.fit(ldim_points)
                y_pred = algorithm.labels_ if hasattr(algorithm, "labels_") else algorithm.predict(ldim_points)
                y_adpred = np.where(y_pred == -1, -1, 1 - y_pred)
                c = Counter(y_pred)
                print('多类簇频率统计：', c)
                wrong = [str(i) for i, pred in enumerate(y_pred) if pred != int(binkey[i])]
                wrong_ad = [str(i) for i, adpred in enumerate(y_adpred) if adpred != int(binkey[i])]
                y = y_pred if len(wrong) <= len(wrong_ad) else y_adpred
                errors = wrong if len(wrong) <= len(wrong_ad) else wrong_ad
                print('聚类错误的操作位：', errors)
                print(name + '结果：', y)
                t = ''.join(['T' if int(binkey[i]) == j else 'F' for i, j in enumerate(y)])
                print(t)
                print('正确聚类个数：' + str(t.count('T')), '错误聚类个数：' + str(t.count('F')))
                t_count_T, t_count_F = t.count('T'), t.count('F')
                iteration_results['labels'].append(y)
                iteration_results['errors'].append(wrong)
                iteration_results['count_T'].append(t_count_T)
                iteration_results['count_F'].append(t_count_F)

                if visualization:
                    plt.subplot(2, len(clustering_algorithms), plot_num)
                    plt.xticks([])
                    plt.yticks([])
                    colors = np.array(
                        list(
                            islice(  # 迭代器切片
                                cycle(
                                    [
                                        "#377eb8",
                                        "#ff7f00",
                                        "#4daf4a",
                                        "#f781bf",
                                        "#a65628",
                                        "#984ea3",
                                        "#999999",
                                        "#e41a1c",
                                        "#dede00",
                                    ]
                                ),
                                int(max(y) + 1),
                            )
                        )
                    )
                    colors = np.append(colors, ["#000000"])
                    if n == 0:
                        plt.title(name, size=18)
                    if vistype == 'cerrs':
                        plt.text(
                            0.99,
                            0.01,
                            t.count('F'),
                            transform=plt.gca().transAxes,
                            size=10,
                            horizontalalignment="right")
                    # n_clusters = int(max(y)) + 1
                    plt.scatter(ldim_points[:, 0], ldim_points[:, 1], s=10, color=colors[y])
                    if vistype == 'cerrs':
                        for index in errors:
                            kk = int(index)
                            plt.scatter(ldim_points[kk, 0],ldim_points[kk, 1], marker="o", edgecolors="r", s=7, facecolors="r")
                    plot_num += 1
            visualization = False
            results.append(iteration_results)

        return iteration_results
