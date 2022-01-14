import sys
import yaml

import numpy as np
from reshaper import Reshaper
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


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
    train_rssis, train_label, test_rssis, test_label = reshaper.get_learnable_single_train_data(train_rate)
    
    # 手法の選択(コマンドライン引数によって決定)
    if method == 'svc':
        # ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
        clf_result = LinearSVC(loss='hinge', random_state=0)
    elif method == 'sgd':
        clf_result = SGDClassifier(loss="hinge")
    elif method == 'kneigh':
        clf_result = KNeighborsClassifier(n_neighbors=knn_neighbors)
        
    train_r = train_rssis[0]
    train_l = train_label[0]
    test_r = test_rssis[0]
    test_l = test_label[0]
    for i in range(1, len(train_rssis)):
        train_r = np.concatenate([train_r, train_rssis[i]], axis=0)
        train_l += train_label[i]
        test_r = np.concatenate([test_r, test_rssis[i]], axis=0)
        test_l += test_label[i]
        
    clf_result.fit(train_r, train_l)
    
    # 予測
    pre = clf_result.predict(test_r)
    # Confusion Matrix出力
    print('---------- Confusion Matrix ----------')
    print(confusion_matrix(test_l, pre))
    print('--------------------------------------')
    # 正答率計算
    ac_score = accuracy_score(test_l, pre)
    print('正答率 =', ac_score)


def error(err_str):
    print(err_str)
    exit()
    

if __name__ == '__main__':
    main()
