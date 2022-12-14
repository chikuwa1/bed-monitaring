import sys
import yaml
import numpy as np
from reshaper import Reshaper
from result_drawer import ResultDrawer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from itertools import combinations
import matplotlib.pyplot as plt


def main():
    with open('tester_name.txt', 'r') as f:
        tester_name = f.read().splitlines()  # 改行文字を消去

    with open('config.yml') as yml:
        config = yaml.safe_load(yml)

    csv_path = config['csv_path']
    data_range = config['data_range']
    knn_neighbors = config['knn_neighbors']
    no_reaction_rssi = config['no_reaction_rssi']
    posture_class_num = config['posture_class_num']
    err_message = f'''usage: python clustering.py <clustering method> <count of train(int:1~{len(tester_name)-1})>

    available clustering method:
        svc     Linear SVC(SVM Classification)
        sgd     SGD(Stochastic Gradient Descent)
        kneigh  K-NeighborsClassifier (K={knn_neighbors})
    '''

    # コマンドライン引数のバリデーション
    if len(sys.argv) != 3:
        error(err_message)
    method = sys.argv[1]
    if method != 'svc' and method != 'sgd' and method != 'kneigh':
        error(err_message)
    try:
        train_count = int(sys.argv[2])
    except ValueError:
        error(err_message)
    if not 1 <= train_count <= (len(tester_name) - 1):
        error(err_message)

    reshaper = Reshaper(data_range, no_reaction_rssi,
                        csv_path, tester_name, posture_class_num)
    # rssis_train, train_label, rssis_test, test_label = reshaper.get_learnable_multi_train_data(
    #     train_count)
    rssis = reshaper.get_rssis()
    print(rssis)


    # # 手法の選択(コマンドライン引数によって決定)
    # if method == 'svc':
    #     # ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
    #     clf_result = LinearSVC(loss='hinge', random_state=0)
    # elif method == 'sgd':
    #     clf_result = SGDClassifier(loss="hinge")
    # elif method == 'kneigh':
    #     clf_result = KNeighborsClassifier(n_neighbors=knn_neighbors)

    # rd = ResultDrawer(rssis_train, train_label)
    # rd.plot_by_posture()
    # moke = [[] for _ in range(posture_class_num)]
    # for data in rssis_train:
    #     for i, d in enumerate(data):
    #         moke[i].append(d)

    # comb = list(combinations([i for i in range(6)], 2))
    # for c in comb:
    #     print(len(train_label))
    #     plt.clf()
    #     plt.close()
    #     plt.xlabel(str(c[0]), fontsize=18, loc="right")
    #     plt.ylabel(str(c[1]), fontsize=18, loc="top")
    #     plt.scatter(moke[c[0]], moke[c[1]], c=train_label, s=10, cmap=plt.cm.coolwarm)
    #     plt.savefig(f"/mnt/c/Users/chiaki/Desktop/result-{str(c[0])}-{str(c[1])}.png")

    
    # 学習
    # clf_result.fit(rssis_train, train_label)
    
    # # 予測
    # pre = clf_result.predict(rssis_test)
    # # Confusion Matrix出力
    # print('---------- Confusion Matrix ----------')
    # print(confusion_matrix(test_label, pre))
    # print('--------------------------------------')
    # # 正答率計算
    # ac_score = accuracy_score(test_label, pre)
    # print("正答率 =", ac_score)


def error(err_str):
    print(err_str)
    exit()


if __name__ == '__main__':
    main()
