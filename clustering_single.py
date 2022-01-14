import sys
import yaml

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
        
    avg_ac_score = 0.0

    for i in range(len(train_rssis)):
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
