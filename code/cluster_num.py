import numpy as np  
import matplotlib.pyplot as plt 
import sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
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
    pca_new = True  
  
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
plt.xlabel('cluster number', fontproperties=font)
plt.savefig('SSE')



plt.figure()
ax = plt.gca()
plt.title('2021-silhouette coefficient', fontproperties=font)
plt.plot(cluster_num, silhouette_co,'ro-')
plt.xlim((2, 14))
plt.ylim((0.15, 0.4))
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
ax.yaxis.get_offset_text().set(fontproperties = 'Times New Roman',size=14) 
plt.xlabel('cluster number', fontproperties=font)
plt.savefig('SC')