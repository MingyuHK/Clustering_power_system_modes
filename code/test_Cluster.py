from OperationClustering import OperationClustering

def run(file_names, line_num, cluaster_num=None):
    OC = OperationClustering(line_num = line_num, cluaster_num = cluaster_num)
    OC.read_data(file_names)
    OC.process()
    OC.plot_distribution()

    return 0


if __name__ == "__main__":

    fileNames = ['jiangsu2021.csv']
    line_num = 2997

    cluaster_num = [3]

    run(fileNames, line_num, cluaster_num)
