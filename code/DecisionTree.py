import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

class DecisionTreeClass:
    def __init__(self, day_num: int, hour_num: int, unit_num: int, line_num: int, inputFileList: list,
                 inputCapFile: list, inputCurFile: list, labelFile: str):
        self.day_num = day_num
        self.hour_num = hour_num
        self.unit_num = unit_num
        self.line_num = line_num
        self.inputFileList = inputFileList
        self.inputCapFile = inputCapFile
        self.inputCurFile = inputCurFile
        self.labelFile = labelFile

    def _read_output_data(self):
        result_list = []
        for file in self.inputFileList:
            filein = open(file, 'r', encoding='utf-8')
            lines = filein.readlines()
            result = []
            for line in lines:
                line = line.strip("\n").strip().split(',')
                result.append(line)
            result_list.append(result)
            filein.close()
        return result_list

    def _read_cap_data(self):
        status = []
        file_status = open(self.inputCapFile[1], 'r', encoding='utf-8')
        status_lines = file_status.readlines()
        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[(u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 28):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * 31 + (u - 1) * 28 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 30):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31) + (u - 1) * 30 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 30):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30 + 31) + (u - 1) * 30 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30 + 31 + 30) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30 + 31 + 30 + 31) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 30):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31) + (u - 1) * 30 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[self.unit_num * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 30):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[
                    self.unit_num * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31) + (u - 1) * 30 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        for day in range(0, 31):
            for u in range(1, self.unit_num + 1):
                status_line = status_lines[
                    self.unit_num * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30) + (u - 1) * 31 + day]
                status_line = status_line.strip("\n")
                status.append(status_line)

        cap = []
        file_cap = open(self.inputCapFile[0], 'r', encoding='utf-8')
        cap_lines = file_cap.readlines()
        for cap_line in cap_lines:
            cap_line = cap_line.strip().split(',')
            cap.append(float(cap_line[8]))

        return status, cap

    def _read_cur_data(self):
        file_WT_cur = open(self.inputCurFile[0], 'r', encoding='utf-8')
        WT_cur_lines = file_WT_cur.readlines()
        WT_cur = []
        WT_index = []
        for WT_cur_line in WT_cur_lines:
            WT_cur_line = WT_cur_line.strip("\n")
            WT_cur_line = WT_cur_line.strip().split(',')
            WT_index_temp = WT_cur_line[2]
            WT_cur.append(WT_cur_line)
            WT_index.append(WT_index_temp)

        file_PV_cur = open(self.inputCurFile[1], 'r', encoding='utf-8')
        PV_cur_lines = file_PV_cur.readlines()
        PV_cur = []
        PV_index = []
        for PV_cur_line in PV_cur_lines:
            PV_cur_line = PV_cur_line.strip("\n")
            PV_cur_line = PV_cur_line.strip().split(',')
            PV_index_temp = PV_cur_line[2]
            PV_cur.append(PV_cur_line)
            PV_index.append(PV_index_temp)

        return WT_cur, WT_index, PV_cur, PV_index

    def read_label(self):
        filein = open(self.labelFile, 'r', encoding='utf-8')
        lines = filein.readlines()
        all_labels = []
        for line in lines:
            line = line.strip("\n")
            all_labels.append(int(float(line)))
        return all_labels

    def get_features(self):
        # define features
        ## cap
        day_cap = []
        ## cur
        day_WT_cur = [0] * self.day_num
        day_PV_cur = [0] * self.day_num
        ## out feature
        out_max = []
        out_min = []
        out_rate_max = []
        out_rate_min = []
        ## wind feature
        wind_max = []
        wind_min = []
        wind_rate_max = []
        wind_rate_min = []
        ## pv feature
        pv_max = []
        pv_min = []
        pv_rate_max = []
        pv_rate_min = []
        ## renew feature
        renew_max = []
        renew_min = []
        renew_rate_max = []
        renew_rate_min = []
        ## load feature
        load_level = []
        load_diff = []
        load_renew_diff = []
        load_renew_out_diff = []
        ## cost
        cost_max = []
        cost_min = []
        carbon_max = []
        carbon_min = []
        ## section
        section_max = []
        section_min = []
        ## line_over
        day_over_max = []
        day_over_min = []

        # call read_cap_data function to read cap_data
        status, cap = self._read_cap_data()
        # call read_cur_data function to read cur_data
        WT_cur, WT_index, PV_cur, PV_index = self._read_cur_data()
        # call read_data function to read data
        result_list = self._read_output_data()

        for day in range(0, self.day_num):
            # cap
            cap_temp = 0
            for unit in range(0, self.unit_num):
                line1 = status[self.unit_num * day + unit].strip().split(',')
                if line1[-1] == "分配停机" or line1[-1] == "检修停机":
                    cap_temp = cap_temp
                else:
                    cap_temp = cap_temp + cap[unit]
            day_cap.append(cap_temp)

            # cur
            index_temp = day + 1
            index_temp1 = str(index_temp)
            if index_temp1 in WT_index:
                index_temp2 = WT_index.index(index_temp1)
                temp1 = WT_cur[index_temp2]
                temp2 = [float(temp1[i]) for i in range(3, 27)]
                day_WT_cur[day] = max(temp2)

            if index_temp1 in PV_index:
                index_temp2 = PV_index.index(index_temp1)
                temp1 = PV_cur[index_temp2]
                temp2 = [float(temp1[i]) for i in range(3, 27)]
                day_PV_cur[day] = max(temp2)

            # index 0 is output data, index 1 is load data
            day_out = result_list[0][day * 10 + 4]
            day_wind = result_list[0][day * 10 + 6]
            day_pv = result_list[0][day * 10 + 7]
            day_load = result_list[1][day]

            out_hour = np.array([float(day_out[i]) for i in range(3, 27)])
            wind_hour = np.array([float(day_wind[i]) for i in range(3, 27)])
            pv_hour = np.array([float(day_pv[i]) for i in range(3, 27)])
            load_hour = np.array([float(day_load[i]) for i in range(2, 26)])
            renew_hour = wind_hour + pv_hour  
            load_renew_hour = load_hour - renew_hour  
            load_renew_out_hour = load_hour - renew_hour - out_hour  

            out_rate = out_hour / load_hour
            wind_rate = wind_hour / load_hour
            pv_rate = pv_hour / load_hour
            renew_rate = renew_hour / load_hour

            # out feature
            out_max.append(max(out_hour))
            out_min.append(min(out_hour))
            out_rate_max.append(max(out_rate))
            out_rate_min.append(min(out_rate))

            # wind feature
            wind_max.append(max(wind_hour))
            wind_min.append(min(wind_hour))
            wind_rate_max.append(max(wind_rate))
            wind_rate_min.append(min(wind_rate))

            # pv feature
            pv_max.append(max(pv_hour))
            pv_min.append(min(pv_hour))
            pv_rate_max.append(max(pv_rate))
            pv_rate_min.append(min(pv_rate))

            # renew feature
            renew_max.append(max(renew_hour))
            renew_min.append(min(renew_hour))
            renew_rate_max.append(max(renew_rate))
            renew_rate_min.append(min(renew_rate))

            # load feature
            load_level.append(sum(load_hour)/24)
            load_diff.append(max(load_hour) - min(load_hour))
            load_renew_diff.append(max(load_renew_hour) - min(load_renew_hour))
            load_renew_out_diff.append(max(load_renew_out_hour) - min(load_renew_out_hour))

            # index 2 is cost, index 3 is carbon
            cost_day = result_list[2][day]
            carbon_day = result_list[3][day]

            cost_hour = np.array([float(cost_day[i]) for i in range(2, 26)])
            carbon_hour = np.array([float(carbon_day[i]) for i in range(2, 26)])
            cost_max.append(max(cost_hour))
            cost_min.append(min(cost_hour))
            carbon_max.append(max(carbon_hour))
            carbon_min.append(min(carbon_hour))

            # section index 4 is section
            section_day = result_list[4][day]
            section_hour = np.array([float(section_day[i]) for i in range(2, 26)])
            section_max.append(max(section_hour))
            section_min.append(min(section_hour))

            # line_over index 5 is line
            over = []
            for j in range(0, self.hour_num):
                k = 0
                for m in range(0, self.line_num):
                    line_day = result_list[5][self.line_num * day + m]
                    line_hour = float(line_day[8 + j])
                    data_lim = float(line_day[36])
                    if abs(line_hour) > data_lim:
                        k = k + 1
                over.append(k / self.line_num)
            day_over_max.append(max(over))
            day_over_min.append(min(over))

        all_features = [
            day_cap, day_WT_cur, day_PV_cur, cost_max, cost_min, carbon_max, carbon_min,
            out_max, out_min, out_rate_max, out_rate_min, wind_max, wind_min, wind_rate_max,
            wind_rate_min, pv_max, pv_min, pv_rate_max, pv_rate_min, renew_max, renew_min,
            renew_rate_max, renew_rate_min, load_level, load_diff, load_renew_diff, load_renew_out_diff,
            section_max, section_min, day_over_max, day_over_min
        ]

        all_samples = []
        for day in range(0, self.day_num):
            sample = []
            for feature in all_features:
                sample.append(feature[day])
            all_samples.append(sample)

        return all_samples

    def DecisionTree(self, all_samples: list, all_labels: list):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(all_samples, all_labels)
        feature_importances = clf.feature_importances_
        print('feature-importance:', feature_importances)
        feature_names = [
            'day_cap', 'WT_cur', 'PV_cur', 'cost_max', 'cost_min', 'carbon_max', 'carbon_min',
            'out_max', 'out_min', 'out_rate_max', 'out_rate_min', 'wind_max', 'wind_min',
            'wind_rate_max', 'wind_rate_min', 'pv_max', 'pv_min', 'pv_rate_max', 'pv_rate_min',
            'renew_max', 'renew_min', 'renew_rate_max', 'renew_rate_min', 'load_level',
            'load_diff', 'load_renew_diff', 'load_renew_out_diff', 'section_max', 'section_min',
            'day_over_max', 'day_over_min'
        ]
        feature_importances = pd.Series(feature_importances, index=feature_names)
        fig, ax = plt.subplots()
        feature_importances.plot.bar(feature_importances)
        plt.ylim((0, 1))
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        fig.savefig('2021_feature_importance')

        # Draw Tree
        plt.figure(dpi=100, figsize=(100, 100))
        tree.plot_tree(clf, filled=True, feature_names=feature_names)
        plt.savefig('2021_cluster')

        X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, random_state=0)
        clf = DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
        print(
            "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                clfs[-1].tree_.node_count, ccp_alphas[-1]
            )
        )

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()

        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig('2021_alpha')

        clf2 = DecisionTreeClassifier(random_state=0, ccp_alpha=0.02)
        clf2.fit(all_samples, all_labels)
        plt.figure(dpi=100, figsize=(100, 100))
        tree.plot_tree(clf2, filled=True, feature_names=feature_names)
        plt.savefig('2021_cluster_cut')

        plt.close('all')