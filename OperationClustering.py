# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:58:55 2018

@author: hqc
"""
#%% 导入包与
#use scikit learn
#basic clustering process
import numpy as np
import sklearn.decomposition

np.seterr(divide='ignore', invalid='ignore')
import time  
import matplotlib.pyplot as plt
from matplotlib import get_backend, is_interactive, interactive
from sklearn import manifold
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\times.ttf", size=14) 

fileNames = []
allResults = []


class OperationClustering:
    def __init__(self, line_num=0, method=1, cluaster_num=None):
        # self.data = None
        # 不同的降维方法，0：pca 1:tsne 2:isomap 3:LargeVis
        self.method = method
        self.dispension = []
        self.chg_freq = []
        self.dev_season = []
        # self.line_num = line_num
        self.allResults = []
        self.cluaster_num = cluaster_num

        self.dayNum = 365
        self.hourNum = 24
        self.lineNum = line_num

        seasons = np.ones(365)
        seasons[0:60] = 2  # 冬
        seasons[60:150] = 0  # 春
        seasons[150:245] = 1  # 夏
        seasons[245:335] = 0  # 秋
        seasons[335:366] = 2  # 冬
        self.seasons = seasons.astype(int)

    def pca(self, data, dim_out=None, existing_e_vector=[]):
        # 固定降维矩阵PCA函数
        if (not dim_out):
            dim_out = data.shape[1]
        if (len(existing_e_vector) > 0):
            e_vector = existing_e_vector
            return np.matmul(data, e_vector), e_vector
        else:
            sig = np.matmul(np.transpose(data), data)
            # np.savetxt('sig', sig)
            print('sig', sig.shape)
            e_value, e_vector = np.linalg.eig(sig)
            print('e_value', e_value)
            print('e_vector', e_vector)
            new_data = np.matmul(data, e_vector)
            importance = np.cumsum(np.abs(e_value))
            key = importance / importance[-1]
            the_key_index = np.where(key > 0.99)[0][0]
            return new_data[:,0:the_key_index], e_vector[:,0:the_key_index]

    def closest_sample(self, data, k):
        # 计算离中心最近的sample
        closest_ind = np.arange(k)
        closest_ind_tmp = []
        D = pairwise_distances(data, metric='euclidean')
        for i in range(k):
            closest_ind_tmp = np.where(D[0:-k, -i - 1] == np.min(D[0:-k, -i - 1]))
            closest_ind[i] = closest_ind_tmp[0][0]
        return closest_ind

    def change_frequency(self, data):
        # 计算变化频率
        freq = 0
        data_diff = data[1::] - data[0:-1]
        freq = np.sum(data_diff != 0)
        return freq

    def deviation_season(self, data):
        # 计算与季节的偏离程度
        dev_season = []
        dev_season_tmp = []
        for i in range(len(set(data[:, 1]))):
            ind = (data[:, 1] == i)
            # print(np.argmax(np.bincount(data[ind, 0])), np.max(np.bincount(data[ind, 0])) / float(len(data[ind, 0])))
            dev_season_tmp.append(np.max(np.bincount(data[ind, 0])))
        dev_season = np.sum(dev_season_tmp) / float(len(data[:, 0]))
        return dev_season

    def euclDistance(self, vector1, vector2):
        # calculate Euclidean distance
        return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))

    def compact_sep_k(self, centroids, clusterLabel, predata):
        k = centroids.shape[0]
        clusterAssment_tmp = np.zeros((clusterLabel.shape[0], 2))
        # print(clusterLabel.shape[0])
        for i in range(clusterLabel.shape[0]):
            clusterAssment_tmp[i, 0] = clusterLabel[i]
            clusterAssment_tmp[i, 1] = self.euclDistance(predata[i], centroids[int(clusterAssment_tmp[i, 0])])

        # 每一类类内距离平均值
        each_cluster_distance = np.zeros(k)
        for i in range(k):
            ind = (clusterAssment_tmp[:, 0] == i)
            each_cluster_distance[i] = np.sum(clusterAssment_tmp[ind, 1])

        # 类内距离平均值的均值
        cluster_distance = np.sum(each_cluster_distance) / clusterLabel.shape[0]

        ## evaluation separation间隔性
        sep_tmp = []
        for i in range(k):
            for j in range(i + 1, k):
                sep_tmp.append(self.euclDistance(centroids[i], centroids[j]))
        sep = np.min(sep_tmp)
        return cluster_distance, sep

    def compact_sep(self, centroids, centerLabel, clusterLabel, predata):
        # 计算紧密型与间隔性指标
        # 要求centroid的标签按0,1,2...k 顺序排列
        k=centroids.shape[0]
        clusterAssment_tmp = np.zeros((clusterLabel.shape[0], 2))
        for i in range(clusterLabel.shape[0]):
            clusterAssment_tmp[i,0] = clusterLabel[i]
            clusterAssment_tmp[i,1] = self.euclDistance(predata[i], centroids[centerLabel==clusterLabel[i]])
        # 每一类类内距离平均值
        each_cluster_distance = np.zeros(k)
        for i in range(k):
            ind = (clusterAssment_tmp[:, 0] == i)
            each_cluster_distance[i] = np.sum(clusterAssment_tmp[ind, 1])

        # 类内距离平均值的均值
        cluster_distance = np.sum(each_cluster_distance) / clusterLabel.shape[0]

        ## evaluation separation间隔性
        sep_tmp = []
        for i in range(k):
            for j in range(i + 1, k):
                sep_tmp.append(self.euclDistance(centroids[i], centroids[j]))
        sep = np.min(sep_tmp)
        return cluster_distance, sep

    def read_data(self, file_names):
        for file_name in file_names:
            dataSet = []
            fileIn = open(file_name, 'r', encoding='UTF-8')
            sign = 1

            for line in fileIn.readlines():
                lineArr = line.strip().split(',')
                if sign == 1:
                    dataSet.append([float(lineArr[i]) for i in range(7, 7 + self.hourNum)])
                else:
                    sign = 1

            dataPreparing = np.asarray(dataSet)
            dataReady = []
            for j in range(self.hourNum):
                for k in range(self.dayNum):
                    dataReady.append(
                        np.concatenate([[j, k], dataPreparing[[k + l * self.dayNum for l in range(self.lineNum)], j]]))

            dataReadyDay_tmp = []
            for i in range(self.dayNum):
                for j in range(self.hourNum):
                    dataReadyDay_tmp.append(dataReady[j * self.dayNum + i])
            dataReadyDay_tmp = np.array(dataReadyDay_tmp)
            dataReadyDay_tmp = np.reshape(dataReadyDay_tmp[:, 2:], (self.dayNum, self.hourNum * self.lineNum))
            dataReadyDay = (dataReadyDay_tmp - np.mean(dataReadyDay_tmp, axis=0)) / 1000

            ## step 2: PCA preprocessing...            # print("step 2: PCA preprocessing...")
            print('dataReadyDay', dataReadyDay)
            my_pca = sklearn.decomposition.PCA(n_components=0.99, copy=False)
            pca_predata_tmp = my_pca.fit_transform(dataReadyDay)
            pca_predata = np.real(pca_predata_tmp)

            if self.method == 0:
                # if pca_new:
                pca_data, ev = self.pca(pca_predata, existing_e_vector=[])
                # else:
                pca_data = np.real(pca_data[:, 0:2])
                low_data = pca_data
            elif self.method == 1:
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate=200)
                data_tsne = tsne.fit_transform(pca_predata)
                low_data = data_tsne
            elif self.method == 2:
                iso = manifold.Isomap(n_neighbors=50, n_components=6)
                data_iso = iso.fit_transform(pca_predata)
                low_data = data_iso


            rDict = {'ini_data':dataReadyDay,'pca_predata': pca_predata, 'data': low_data}
            self.allResults.append(rDict)

    def process(self):
        for idx_temp in range(len(self.allResults)):

            pca_predata = self.allResults[idx_temp]['pca_predata']
            low_data = self.allResults[idx_temp]['data']

            k = self.cluaster_num[idx_temp]
            kmeans = KMeans(n_clusters=k)
            y_pred = kmeans.fit_predict(pca_predata)
            centers = kmeans.cluster_centers_

            # np.savetxt('pca_data.txt', pca_predata)
            temp = np.matmul(np.transpose(pca_predata), pca_predata)
            dispension_temp = np.trace(np.matmul(np.transpose(pca_predata), pca_predata)) / self.dayNum
            temp1 = np.mean(temp.diagonal())
            dispension_temp = dispension_temp / temp1
            self.dispension.append(np.trace(np.matmul(np.transpose(pca_predata), pca_predata)) / self.dayNum)
            dispension = self.dispension
            print('dispension:', dispension)
            print('dispension_temp:', dispension_temp)
            self.dispension.append(np.trace(np.matmul(np.transpose(pca_predata), pca_predata)) / self.dayNum)
            dispension = self.dispension
            print('dispension:', dispension)
            self.chg_freq.append(self.change_frequency(y_pred))
            change_freq = self.chg_freq
            print('change_freq:', change_freq)
            self.dev_season.append(self.deviation_season(np.c_[y_pred, self.seasons]))
            dev_season = self.dev_season
            print('dev_season:', dev_season)

            ##  find the sample closest to center
            data_cen = np.r_[pca_predata, centers]
            data_cen_ind = self.closest_sample(data_cen, len(centers))
            print('data_cen_ind:', data_cen_ind)
            centers_sample = low_data[data_cen_ind, :]
            centers_label = kmeans.predict(pca_predata[data_cen_ind, :])

            ## 下面修正标签，保证冬季的标签为0
            first_label = y_pred[0]
            if (first_label != 0):
                y_pred[y_pred == first_label] = k
                y_pred[y_pred == 0] = first_label
                y_pred[y_pred == k] = 0
                centers_label[centers_label == first_label] = k
                centers_label[centers_label == 0] = first_label
                centers_label[centers_label == k] = 0
            self.allResults[idx_temp].update({'y_pred': y_pred, 'method': self.method, 'centers': centers,
                     'centers_sample': centers_sample, 'centers_label': centers_label,
                     'clustering': 'KMeans'})
            print('y_pred:', y_pred)
            np.savetxt('low_data.txt', low_data)
            np.savetxt('y_pred.txt', y_pred)
            # cluster_distance_tmp, sep_tmp = self.compact_sep_k(centroids, y_label, predata)
            cluster_distance_tmp, sep_tmp = self.compact_sep_k(centers, y_pred, pca_predata)
            print('distance;', cluster_distance_tmp)
        # return 0

    def plot_distribution(self):
        # %% 根据方差对低维数据进行放缩
        var_scale = self.dispension / self.dispension[-1]
        # var_scale = cluster_distance/cluster_distance[2]
        for i in range(len(self.allResults)):
            self.allResults[i]['data'] = self.allResults[i]['data'] * var_scale[i]

        # %% print allResults 聚类标签
        font = FontProperties(fname=r"c:\windows\fonts\times.ttf", size=24)
        colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan',  'brown', 'peru', 'sage', 'teal', 'pink'])
        markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
        title = [u'2021 (9%)']
        
        axis_distance = [40, 40, 40]
        for i in range(len(self.allResults)):
            low_data = self.allResults[i]['data']
            y_pred = self.allResults[i]['y_pred']
            centers_sample = self.allResults[i]['centers_sample']
            centers_label = self.allResults[i]['centers_label']

            with plt.style.context('seaborn-white'):
                plt.figure()
                # print(len(low_data))
                for j in range(len(low_data)):
                    plt.scatter(low_data[j, 0], low_data[j, 1], c=colors[y_pred[j]], marker=markers[y_pred[j]],
                                edgecolors='k',
                                linewidths=1, alpha=1)
                plt.xlabel(u'First principal feature', fontproperties=font)
                plt.ylabel(u'Second principal feature', fontproperties=font)
                plt.title(title[i], fontproperties=font)
                plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
                plt.savefig(title[i] + u'散点图', dpi=400)

        # %% print allResults 时序作图
        colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan', 'brown', 'peru', 'sage', 'teal', 'pink'])
        markers = ['_', 'x', '+', 'v', 's', 'p', '*', '_', '_', '_', '_']
        # 输出表格的标题
        # here is the name of title
        for i in range(len(self.allResults)):
            low_data = self.allResults[i]['data']
            x_date = np.arange(1, 366)
            y_pred = self.allResults[i]['y_pred']
            centers_sample = self.allResults[i]['centers_sample']
            centers_label = self.allResults[i]['centers_label']
            with plt.style.context('seaborn-white'):
                plt.figure()
                for j in range(len(low_data)):
                    plt.scatter(x_date[j], int(y_pred[j]), c=colors[y_pred[j]], edgecolors='k', linewidths=0.1, alpha=1)
                plt.plot(x_date, y_pred, 'k', linewidth=0.5, alpha=1)
                plt.xlabel(u'Day', fontproperties=font)
                plt.ylabel(u'Cluster', fontproperties=font)
                # 设定刻度范围
                plt.yticks(np.arange(0, self.cluaster_num[i], 1))
                plt.title(title[i], fontproperties=font)
                plt.xlim((1, 365))
                plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
                plt.savefig(title[i] + u'时序图', dpi=400)