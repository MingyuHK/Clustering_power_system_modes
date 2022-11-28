# from labels to get the mean value of each cluster
import numpy as np

y_pred = np.loadtxt('y_pred.txt', dtype=int)

filein = open('2021_output.csv', 'r', encoding='utf-8')
lines = filein.readlines()
output = []
for line in lines:
    line = line.strip("\n").strip().split(',')
    output.append(line)
filein.close()

filein = open('2021_load.csv', 'r', encoding='utf-8')
lines = filein.readlines()
load = []
for line in lines:
    line = line.strip("\n").strip().split(',')
    load.append(line)
filein.close()

output_cluster = []
for cluster in range(0, 3):
    day_cluster = np.argwhere(y_pred==cluster)
    print(len(day_cluster))
    day_cluster = day_cluster.flatten()
    feature_list = []
    for feature in range(0, 10):
        day_feature_sum = 0
        for day in day_cluster:
            day_feature = output[day*10 + feature]
            day_feature = np.array([float(day_feature[i]) for i in range(3, 27)])
            day_feature_sum = day_feature_sum + day_feature
        feature_mean = day_feature_sum/len(day_cluster)
        feature_list.append(feature_mean)

    load_sum = 0
    for day in day_cluster:
        day_load = load[day]
        day_load = np.array([float(day_load[i]) for i in range(2, 26)])
        load_sum = load_sum + day_load
    load_mean = load_sum/len(day_cluster)
    feature_list.append(load_mean)

    name ='cluster'+str(cluster)+'.csv'   
    np.savetxt(name, feature_list,delimiter=",")
        





