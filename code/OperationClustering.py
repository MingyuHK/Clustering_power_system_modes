import numpy as np
import sklearn.decomposition

np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib import get_backend, is_interactive, interactive
from sklearn import manifold
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\\fonts\\times.ttf", size=14) 

fileNames = []
allResults = []


class OperationClustering:
    def __init__(self, line_num=0, cluaster_num=None):

        self.dayNum = 365
        self.hourNum = 24
        self.lineNum = line_num
        self.allResults = []
        self.cluaster_num = cluaster_num

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

            ## step 2: PCA preprocessing...            
            my_pca = sklearn.decomposition.PCA(n_components=0.99, copy=False)
            pca_predata_tmp = my_pca.fit_transform(dataReadyDay)
            pca_predata = np.real(pca_predata_tmp)

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate=200)
            data_tsne = tsne.fit_transform(pca_predata)
            low_data = data_tsne

            rDict = {'ini_data':dataReadyDay,'pca_predata': pca_predata, 'data': low_data}
            self.allResults.append(rDict)

    def process(self):
        for idx_temp in range(len(self.allResults)):

            pca_predata = self.allResults[idx_temp]['pca_predata']

            k = self.cluaster_num[idx_temp]
            kmeans = KMeans(n_clusters=k)
            y_pred = kmeans.fit_predict(pca_predata)
            centers = kmeans.cluster_centers_
            self.allResults[idx_temp].update({'y_pred': y_pred,'centers': centers,'clustering': 'KMeans'})

            np.savetxt('y_pred.txt', y_pred)

    def plot_distribution(self):
       
        # %% print allResults 
        font = FontProperties(fname=r"c:\windows\\fonts\\times.ttf", size=24)
        colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan',  'brown', 'peru', 'sage', 'teal', 'pink'])
        markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
        title = [u'2021 (9%)']
        
        for i in range(len(self.allResults)):
            low_data = self.allResults[i]['data']
            y_pred = self.allResults[i]['y_pred']
            with plt.style.context('seaborn-white'):
                plt.figure()
                for j in range(len(low_data)):
                    plt.scatter(low_data[j, 0], low_data[j, 1], c=colors[y_pred[j]], marker=markers[y_pred[j]],
                                edgecolors='k',
                                linewidths=1, alpha=1)
                plt.xlabel(u'First principal feature', fontproperties=font)
                plt.ylabel(u'Second principal feature', fontproperties=font)
                plt.title(title[i], fontproperties=font)
                plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
                plt.savefig(title[i] + u'Scatter_plot', dpi=400)

        # %% print allResults 
        colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan', 'brown', 'peru', 'sage', 'teal', 'pink'])
        markers = ['_', 'x', '+', 'v', 's', 'p', '*', '_', '_', '_', '_']
        # here is the name of title
        for i in range(len(self.allResults)):
            low_data = self.allResults[i]['data']
            x_date = np.arange(1, 366)
            y_pred = self.allResults[i]['y_pred']
            with plt.style.context('seaborn-white'):
                plt.figure()
                for j in range(len(low_data)):
                    plt.scatter(x_date[j], int(y_pred[j]), c=colors[y_pred[j]], edgecolors='k', linewidths=0.1, alpha=1)
                plt.plot(x_date, y_pred, 'k', linewidth=0.5, alpha=1)
                plt.xlabel(u'Day', fontproperties=font)
                plt.ylabel(u'Cluster', fontproperties=font)
                plt.yticks(np.arange(0, self.cluaster_num[i], 1))
                plt.title(title[i], fontproperties=font)
                plt.xlim((1, 365))
                plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
                plt.savefig(title[i] + u'Time_plot', dpi=400)