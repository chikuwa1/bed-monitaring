# プログラムを理解するためにメモする用のダミープログラム

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

    csv_path = config['csv_path']# CSVファイルのパス= ./csv/log1208/
    data_range = config['data_range']# データ整形時に平均値を取るデータの範囲=50
    knn_neighbors = config['knn_neighbors']# K近傍法の近傍オブジェクト数=3
    no_reaction_rssi = config['no_reaction_rssi']# 無反応のセンサーに対する代替のRSSI値=-105.0
    posture_class_num = config['posture_class_num']# 姿勢クラス数=7
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
    # data_range=50, no_reaction_rssi=-105.0, csv_path=./csv/log1208/, posture_class_num=7 (48行目付近)
    # tester_name は11名の名前が入ったリスト(39,83行目)
    # これらをReshaperへ引き渡す
    train_rssis, train_label, test_rssis, test_label = reshaper.get_learnable_single_train_data(train_rate)
    # Reshaperのget_learnable_multi_train_data関数を用いて訓練用，テスト用のRSSIとラベルを代入

    ## 手法の選択(コマンドライン引数によって決定)
    if method == 'svc':
        # ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
        clf_result = LinearSVC(loss='hinge', random_state=0)
    elif method == 'sgd':
        clf_result = SGDClassifier(loss="hinge")
    elif method == 'kneigh':
        clf_result = KNeighborsClassifier(n_neighbors=knn_neighbors)
        
    avg_ac_score = 0.0

    #被験者ごとラベル
    for i in range(len(train_rssis)): # 11回繰り返されている
        dir_path_human = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/'
        if not os.path.exists(dir_path_human):
            os.mkdir(dir_path_human)
            # subprocess.run(['mkdir', dir_path])
        else:
            shutil.rmtree(dir_path_human)
            os.mkdir(dir_path_human)
            # subprocess.run(['mkdir', dir_path])
        posture_rssis = [[] for _ in range(posture_class_num)] # 姿勢クラス数=7 -> range(7)=0,1,2,3,4,5,6
        # posture_rssis[7][?]の状態
        for data in train_rssis[i]: # 11人分の訓練用RSSIを1人ずつdataに入れてる
            for j, d in enumerate(data): # jには1から順に数値を，dにはdata（1人分のRSSI）が入る
                posture_rssis[j].append(d) # １人分の姿勢クラスごとにRSSIを追加
        label_comb = list(combinations([i for i in range(6)], 2)) # 0~5で２通りずつリストに追加 list[(0,1), (0,2), (0,3),...,(4,5)]の15個
        for comb in label_comb:
            plt.clf()
            plt.close()
            
            plt.xlabel(str(comb[0]), fontsize=18, loc="right")
            plt.ylabel(str(comb[1]), fontsize=18, loc="top")
            

            # plt.scatter(posture_rssis[c[0]], posture_rssis[c[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm, label=train_label[i])
            plt.scatter(posture_rssis[comb[0]], posture_rssis[comb[1]], c=train_label[i], s=10, cmap=plt.cm.coolwarm)
            # plt.legend() 
            png_path = f'/mnt/c/Users/chiaki/Desktop/human{str(i)}/human{str(i)}-{str(comb[0])}-{str(comb[1])}.png'           
            plt.savefig(png_path)

    
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
