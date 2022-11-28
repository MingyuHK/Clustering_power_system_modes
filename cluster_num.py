# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:05:51 2018

@author: hqc
"""

#%% 导入包与
#use scikit learn
#basic clustering process
import numpy as np  
import time  
import matplotlib.pyplot as plt 
import sklearn.decomposition
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\\fonts\\times.ttf", size=20)

fileNames = ['jiangsu2021.csv']
k_cluster_distance=[]
k_sep=[]
k_sil_s=[]
for fileName in fileNames:  
    dataSet = [] 
    dayNum = 365
    hourNum = 24
    lineNum = 2997
    fileIn = open(fileName, 'r', encoding='UTF-8')
    sign = 1
    pca_new = True       #True表示首次pca，False表示用前面变化得到的ev矩阵
  
    for line in fileIn.readlines():  
        lineArr = line.strip().split(',')
        if(sign==1):
            dataSet.append([float(lineArr[i]) for i in range(7, 7+hourNum)])
        else:
            sign = 1

    print(np.shape(dataSet))

    dataPreparing = np.asarray(dataSet)

    print(np.shape(dataPreparing))

    dataReady = []
    for j in range(hourNum):
        for k in range(dayNum):
            dataReady.append(
                np.concatenate([[j,k],dataPreparing[[k+l*dayNum for l in range(lineNum)],j]]))
        
    dataReadyDay_tmp = []
    for i in range(dayNum):
        for j in range(hourNum):
            dataReadyDay_tmp.append(dataReady[j*dayNum+i])
    dataReadyDay_tmp = np.array(dataReadyDay_tmp)
    dataReadyDay_tmp = np.reshape(dataReadyDay_tmp[:,2:],(dayNum,hourNum*lineNum))
    #中心化
    dataReadyDay = (dataReadyDay_tmp-np.mean(dataReadyDay_tmp,axis=0))/500

    my_pca = sklearn.decomposition.PCA(n_components=0.99, copy=False)
    pca_predata_tmp = my_pca.fit_transform(dataReadyDay)
    pca_predata = np.real(pca_predata_tmp)

sse = []
silhouette_co = []
DBI = []
for k in range(2,15):
        print(str(k) + " clustering...")
        print(pca_predata.shape)
        kmeans = KMeans(n_clusters=k).fit(pca_predata)
        # y_pred = kmeans.fit_predict(pca_predata)
        sse.append(kmeans.inertia_)
        score_sc = silhouette_score(pca_predata, kmeans.labels_)
        silhouette_co.append(score_sc)
        score_dbi = davies_bouldin_score(pca_predata, kmeans.labels_)
        DBI.append(score_dbi)

rnw_type=u'江苏'

cluster_num = np.arange(2, 15)  
plt.figure()
ax = plt.gca()
plt.title('2021-compactness', fontproperties=font)
plt.plot(cluster_num, sse,'ro-')
plt.xlim((2, 14))
plt.ylim((1.5e6, 4.5e6))
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
ax.yaxis.get_offset_text().set(fontproperties = 'Times New Roman',size=14) 
# plt.legend(prop={'family':'SimHei','size':12})
plt.xlabel('cluster number', fontproperties=font)
# plt.savefig('SSE.svg', format='svg')
plt.savefig('C:\\Users\\eee\\Desktop\\fig\\SSE.pdf')



plt.figure()
ax = plt.gca()
plt.title('2021-silhouette coefficient', fontproperties=font)
plt.plot(cluster_num, silhouette_co,'ro-')
plt.xlim((2, 14))
plt.ylim((0.15, 0.4))
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
ax.yaxis.get_offset_text().set(fontproperties = 'Times New Roman',size=14) 
# plt.legend(prop={'family':'SimHei','size':12})
plt.xlabel('cluster number', fontproperties=font)
plt.savefig('C:\\Users\\eee\\Desktop\\fig\\SC.pdf')

# plt.figure()
# plt.title(rnw_type+u'-DBI', fontproperties=font)
# plt.plot(range(2,15), DBI,'ro-',label=u'2021')
# plt.legend(prop={'family':'SimHei','size':12})
# plt.xlabel(u'类别数量', fontproperties=font)
# plt.savefig('DBI')

# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(2, 15), timings=False, locate_elbow=False)
# visualizer.fit(pca_predata)
# visualizer.show(outpath="SSE.png")

# model = KMeans()
# visualizer1 = KElbowVisualizer(model, k=(2, 15), metric='calinski_harabasz', timings=False)
# visualizer1.fit(pca_predata)
# visualizer1.show(outpath="CH.png")

# model = KMeans()
# visualizer2 = KElbowVisualizer(model, k=(2, 15), metric='silhouette', timings=False)
# visualizer2.fit(pca_predata)
# visualizer2.show(outpath="SC.png")

# model1 = KMeans(5)
# visualizer = SilhouetteVisualizer(model1)
# visualizer.fit(pca_predata)    # Fit the data to the visualizer
# visualizer.show(outpath="SC5.png")    # Finalize and render the figure

# model = KMeans(5)
# visualizer = InterclusterDistance(model)
# visualizer.fit(pca_predata)        # Fit the data to the visualizer
# visualizer.show(outpath="ID5.png")        # Finalize and render the figure