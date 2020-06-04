import numpy as np
import os
import sys

# how to use : python MyClassifier.py training.csv testing.csv 10NN


# file_name = "pima.csv"
# csv_data = pd.read_csv(file_name, header=None)  # 读取训练数据
# print(csv_data.shape)  # (768,9)
# print(csv_data[8])

# if len(sys.argv) != 4:
# print("argv number error")
# os._exit(0)

# for i in range(0, len(sys.argv)):
# print("argv " + str(i) + " " + str(sys.argv[i]))

def read(filename):
    file = open(filename, 'r')
    data = []
    for line in file.readlines():
        line_data = line.strip().split(',')
        data.append(line_data)
    file.close()
    data = np.array(data)
    # print(data.shape)
    return data


def split_data(data):
    label = data[:, -1]
    # print(label)
    train = np.delete(data, -1, axis=1)
    # print(train)
    return train, label


def read_traindata(filename):
    data = read(filename)
    return split_data(data)


class KNN(object):
    def __init__(self, n=5):
        self.num_of_neighbors = n
        self.x = []
        self.y = []
        self.distances = []
        self.result = []

    def train(self, train_data, train_label):
        self.x = np.array(train_data)
        self.y = np.array(train_label)

    def predict(self, test_data):
        self.result.clear()
        # 对每一个样本求预测结果
        for i in range(len(test_data)):
            # 欧式距离
            self.distances = np.sum(np.square(self.x - test_data[i]), axis=1)
            self.distances = self.distances ** 0.5

            # 距离排序从小到大，得到下标
            index = np.argsort(self.distances)

            # 统计每个类别出现的次数
            y_count = {}
            for j in range(self.num_of_neighbors):
                y_possible = self.y[index[j]]
                y_count[y_possible] = y_count.get(y_possible, 0) + 1

            # 找出出现次数最大的类别
            y_predict = max(y_count, key=y_count.get)

            self.result.append(y_predict)
        return self.result




def main():
    ##test data
    # train_data = np.array([[1,1,1],[2,2,2],[1,1,1.1],[1,1,1.1],[2.1,1.1,1.1],[1,2,1.1]])
    # train_label = np.array(['one', 'two', 'one', 'one', 'two', 'two'])
    # test_data = np.array([[1.0,1.0,1.0], [2, 1.9, 1.8]])

    ##kkn test
    # knn = KNN()
    # knn.train(train_data, train_label)
    # print(knn.predict(test_data))


    # file_name = "pima.csv"

    # csv_data = pd.read_csv(file_name, header=None)  # 读取训练数据
    # print(csv_data.shape)  # (768,9)
    # print(csv_data[8])

    if len(sys.argv) != 4:
        print("argv number error")
        os._exit(0)

    # for i in range(0, len(sys.argv)):
    # print("argv " + str(i) + " " + str(sys.argv[i]))

    train_file = "./" + sys.argv[1]
    test_file = "./" + sys.argv[2]

    train_data, train_label = read_traindata(train_file)
    train_data = train_data.astype(float)
    test_data = read(test_file).astype(float)

    if sys.argv[3][-1] == "N" and sys.argv[3][-2] == "N":
        k = int(sys.argv[3][-3])
        knn = KNN(k)
        knn.train(train_data, train_label)
        result = knn.predict(test_data)
        for i in result:
            print(i)

    # count = 0
    # for i in range(len(test_data)):
    #     if result[i] == train_label[i]:
    #         count += 1
    # print(count / len(test_data))


if __name__ == "__main__":
    # test read
    # data = read("./pima.csv")

    # test data_split
    # split_data(data)

    main()
