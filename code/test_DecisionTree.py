from DecisionTree import DecisionTreeClass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    day_num = 365
    hour_num = 24
    unit_num = 156
    line_num = 2709
    inputFileList = ['2021_output.csv', '2021_load.csv', '2021cost.csv', '2021carbon.csv', '2021section.csv',
                     '2021line_raw.csv']
    inputCapFile = ['2021_cap.csv', '2021_status.csv']
    inputCurFile = ['2021WT_cur.csv', '2021PV_cur.csv']
    labelFile = 'y_pred.txt'

    DTC = DecisionTreeClass(day_num, hour_num, unit_num, line_num, inputFileList,
                            inputCapFile, inputCurFile, labelFile)
    all_samples = DTC.get_features()

    all_samples_df = pd.DataFrame(all_samples)
    all_samples_df.to_csv('all_samples.csv', sep=',', encoding='utf-8', header=False, index=False)
    new_all_samples_df = all_samples_df.corr(method='spearman')
    plt.figure(1)
    sns.heatmap(new_all_samples_df, vmax=1, vmin=-1)  # 绘制new_df的矩阵热力图
    print(new_all_samples_df.iloc[21, 4])
    plt.savefig('2021_relates')
    plt.close('all')
    all_labels = DTC.read_label()
    DTC.DecisionTree(all_samples, all_labels)


