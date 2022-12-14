import sys
import yaml
import os
import shutil

from reshaper import Reshaper
# from result_drawer import
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from itertools import combinations
# import subprocess
import numpy as np


def main():
    with open('tester_name.txt', 'r') as f:
        testers = f.read().splitlines()  # 改行文字を消去

    with open('config.yml') as yml:
        config = yaml.safe_load(yml)

    csv_path = config['csv_path']
    data_range = config['data_range']
    knn_neighbors = config['knn_neighbors']
    no_reaction_rssi = config['no_reaction_rssi']
    posture_class_num = config['posture_class_num']
    err_message = f'''usage: python clustering.py <clustering method> <train rate(0<rate<1)>

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

    train_rate = float(sys.argv[2])
    if not 0 < train_rate < 1:
        error(err_message)

    reshaper = Reshaper(data_range, no_reaction_rssi,
                        csv_path, testers, posture_class_num)
    train_rssis, train_label, test_rssis, test_label = reshaper.get_learnable_single_train_data(
        train_rate)

    # 手法の選択(コマンドライン引数によって決定)
    if method == 'svc':
        # ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
        clf_result = LinearSVC(loss='hinge', random_state=0)
    elif method == 'sgd':
        clf_result = SGDClassifier(loss="hinge")
    elif method == 'kneigh':
        clf_result = KNeighborsClassifier(n_neighbors=knn_neighbors)

    avg_ac_score = 0.0

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i in range(posture_class_num):
        dir_path_posture = f'/mnt/c/Users/chiaki/Desktop/posture{str(i)}/'
        if not os.path.exists(dir_path_posture):
            os.mkdir(dir_path_posture)
        else:
            shutil.rmtree(dir_path_posture)
            os.mkdir(dir_path_posture)
        human_rssis = [[] for _ in range(posture_class_num)]
        

    for i, data_per_human in enumerate(train_rssis):
        dir_path_human = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/'
        if not os.path.exists(dir_path_human):
            os.mkdir(dir_path_human)
        else:
            shutil.rmtree(dir_path_human)
            os.mkdir(dir_path_human)
        posture_rssis = [[] for _ in range(6)]
        for data in data_per_human:
            for j, d in enumerate(data):
                posture_rssis[j].append(d)
        comb = list(combinations([i for i in range(6)], 2))
        prev_label = 0
        new_label_start_idx = 0
        rssis_idx_per_label = []

        for j, l in enumerate(train_label[i]):
            if prev_label != l:
                idx_range = (new_label_start_idx, j-1)
                rssis_idx_per_label.append(idx_range)
                prev_label = l
                new_label_start_idx = j
        idx_range = (new_label_start_idx, len(train_label[i]))
        rssis_idx_per_label.append(idx_range)

        print(rssis_idx_per_label)

        for c in comb:
            plt.clf()
            plt.close()

            plt.xlabel(str(c[0]), fontsize=18, loc="right")
            plt.ylabel(str(c[1]), fontsize=18, loc="top")

            # plt.scatter(posture_rssis[c[0]], posture_rssis[c[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm, label=train_label[i])
            # plt.scatter(posture_rssis[c[0]], posture_rssis[c[1]],
            #             c=train_label[i], s=10, cmap=plt.cm.coolwarm)

            for j, ripl in enumerate(rssis_idx_per_label):
                plt.scatter(posture_rssis[c[0]][ripl[0]:ripl[1]], posture_rssis[c[1]]
                            [ripl[0]:ripl[1]], c=colors[j], marker='D', alpha=0.05, label=f'posture {j}')
            # plt.legend()
            plt.legend() # (7)凡例表示
            png_path = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/human{str(i)}-{str(c[0])}-{str(c[1])}.png'
            plt.savefig(png_path)
        


    # 被験者ごとラベル
    # for i in range(len(train_rssis)):
    #     dir_path_human = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/'
    #     if not os.path.exists(dir_path_human):
    #         os.mkdir(dir_path_human)
    #         # subprocess.run(['mkdir', dir_path])
    #     else:
    #         shutil.rmtree(dir_path_human)
    #         os.mkdir(dir_path_human)
    #         # subprocess.run(['mkdir', dir_path])
    #     posture_rssis = [[] for _ in range(posture_class_num)]
    #     print(len(train_rssis[0][0]))
    #     for data in train_rssis[i]:
    #         for j, d in enumerate(data):
    #             posture_rssis[j].append(d)
    #     comb = list(combinations([i for i in range(6)], 2))
    #     for c in comb:
    #         plt.clf()
    #         plt.close()

    #         plt.xlabel(str(c[0]), fontsize=18, loc="right")
    #         plt.ylabel(str(c[1]), fontsize=18, loc="top")

    #         # plt.scatter(posture_rssis[c[0]], posture_rssis[c[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm, label=train_label[i])
    #         plt.scatter(posture_rssis[c[0]], posture_rssis[c[1]],
    #                     c=train_label[i], s=10, cmap=plt.cm.coolwarm)
    #         # plt.legend()
    #         png_path = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/human{str(i)}-{str(c[0])}-{str(c[1])}.png'
    #         plt.savefig(png_path)

    # 姿勢ごとのラベル

    # pouse_data = [[[] for _ in range(posture_class_num)] for _ in range(len(testers))]
    # rssis_np = np.array(reshaper.get_rssis())
    # posture_data_np = rssis_np.T
    # posture_data = posture_data_np.tolist()

    # for i in range(posture_class_num):
    #     dir_path_posture = f'/mnt/c/Users/chiaki/Desktop/posture{str(i)}/'
    #     if not os.path.exists(dir_path_posture):
    #         os.mkdir(dir_path_posture)
    #     else:
    #         shutil.rmtree(dir_path_posture)
    #         os.mkdir(dir_path_posture)
    #         #subprocess.run(['mkdir', f'/mnt/c/Users/chiaki/Desktop/posture{str(i)}/'])

    #     moke = [[] for _ in range(len(testers))]
    #     for j in range(len(testers)):
    #         moke[j].append(posture_data[i][j])
    #     comb = list(combinations([i for i in range(6)], 2))
    #     for c in comb:
    #         plt.clf()
    #         plt.close()

    #         plt.xlabel(str(c[0]), fontsize=18, loc="right")
    #         plt.ylabel(str(c[1]), fontsize=18, loc="top")

    #         # plt.scatter(moke[c[0]], moke[c[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm, label=train_label[i])
    #         plt.scatter(moke[c[0]], moke[c[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm)
    #         # plt.legend()
    #         png_path = f'/mnt/c/Users/chiaki/Desktop/posture{str(i)}/posture0-{str(c[0])}-{str(c[1])}.png'
    #         plt.savefig(png_path)

    # 学習
    clf_result.fit(train_rssis[i], train_label[i])
    # 予測
    pre = clf_result.predict(test_rssis[i])
    # Confusion Matrix出力
    print('---------- Confusion Matrix ----------')
    print(confusion_matrix(test_label[i], pre))
    print('--------------------------------------')
    # 正答率計算
    ac_score = accuracy_score(test_label[i], pre)
    avg_ac_score += ac_score
    print('正答率 =', ac_score)

    print('平均正解率 =', avg_ac_score / len(testers))


def error(err_str):
    print(err_str)
    exit()


if __name__ == '__main__':
    main()
