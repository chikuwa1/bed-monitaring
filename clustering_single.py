import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from reshaper import Reshaper

DATA_RANGE = 50             # 平均値をとるデータの個数
NO_REACTION_RSSI = -105.0   # 無反応のデータに対する代替のRSSI値
CSV_PATH = './csv/log1208/' # CSVファイルの場所
NEIGHBORS = 3               # K近傍法のK値
CLASS_NUM = 7               # 姿勢クラス数

ARGS_ERROR = f'''usage: python clustering.py <clustering method> <train rate(0<rate<1)>

available clustering method:
    svc     Linear SVC(SVM Classification)
    sgd     SGD(Stochastic Gradient Descent)
    kneigh  K-NeighborsClassifier (K={NEIGHBORS})
'''
"""実行時，コマンドライン引数に誤りがあった場合のエラー文
"""

def main():
    # コマンドライン引数のバリデーション
    if len(sys.argv) != 3:
        print(ARGS_ERROR)
        exit()
    method = sys.argv[1]
    if method != 'svc' and method != 'sgd' and method != 'kneigh':
        print(ARGS_ERROR)
        exit()

    with open('tester_name.txt', 'r') as f:
        testers = f.read().splitlines() # 改行文字を消去

    train_rate = float(sys.argv[2])
    if not 0 < train_rate < 1:
        print(ARGS_ERROR)
        exit()

    reshaper = Reshaper(DATA_RANGE, NO_REACTION_RSSI,
                        CSV_PATH, testers, CLASS_NUM)
    train_rssis, train_label, test_rssis, test_label = reshaper.get_learnable_single_train_data(train_rate)
    
    # 手法の選択(コマンドライン引数によって決定)
    if method == 'svc':
        print('LinearSVC classifing...  ', end='')
        # ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
        clf_result = LinearSVC(loss='hinge', random_state=0)
    elif method == 'sgd':
        print('SGD classifing...        ', end='')
        clf_result = SGDClassifier(loss="hinge")
    elif method == 'kneigh':
        print('KNeighbors classifing...        ', end='')
        clf_result = KNeighborsClassifier(n_neighbors=NEIGHBORS)
        
    avg_ac_score = 0.0

    for i in range(len(train_rssis)):
        # 学習
        clf_result.fit(train_rssis[i], train_label[i])
        
        # 予測
        pre = clf_result.predict(test_rssis[i])
        print('\033[32m'+'Done'+'\033[0m')

        # Confusion Matrix出力
        print('---------- Confusion Matrix ----------')
        print(confusion_matrix(test_label[i], pre))
        print('--------------------------------------')

        # 正答率計算
        ac_score = accuracy_score(test_label[i], pre)
        avg_ac_score += ac_score
        print('正答率 =', ac_score)
        
    print('平均正解率 =', avg_ac_score / len(testers))

if __name__ == '__main__':
    main()
