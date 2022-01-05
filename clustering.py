import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from reshaper import Reshaper

DATA_RANGE = 50             # 平均値をとるデータの個数
NO_REACTION_RSSI = -105.0   # 無反応のデータに対する代替のRSSI値
CSV_PATH = './csv/log1208/' # CSVファイルの場所
NEIGHBORS = 3               # K近傍法のK値
CLASS_NUM = 7               # 姿勢クラス数
TESTER_NUM = 11             # 被験者の人数

ARGS_ERROR = f'''usage: python clustering.py <clustering method> <num of tester(int:1~{TESTER_NUM-1})>

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
    try:
        tester_num = int(sys.argv[2])
    except ValueError:
        print(ARGS_ERROR)
        exit()
    if not 1 <= tester_num <= (TESTER_NUM - 1):
        print(ARGS_ERROR)
        exit()

    with open('tester_name.txt', 'r') as f:
        tester_name = f.read().splitlines() # 改行文字を消去

    reshaper = Reshaper(DATA_RANGE, NO_REACTION_RSSI,
                        CSV_PATH, tester_name, CLASS_NUM)
    avged_rssis, rssi_classes = reshaper.parse_avged_rssi_and_cls()
    rssis_train, train_label, rssis_test, test_label = make_learnable_data(avged_rssis, rssi_classes, tester_num)
    
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

    # 学習
    clf_result.fit(rssis_train, train_label)
    
    # 予測
    pre = clf_result.predict(rssis_test)
    print('\033[32m'+'Done'+'\033[0m')

    # Confusion Matrix出力
    print('---------- Confusion Matrix ----------')
    print(confusion_matrix(test_label, pre))
    print('--------------------------------------')

    # 正答率計算
    ac_score = accuracy_score(test_label, pre)
    print("正答率 =", ac_score)


def make_learnable_data(avged_rssis, rssi_classes, tester_num):
    # 標準化
    rssi_all = []
    for rssi in avged_rssis:
        rssi_all += rssi
    sc = preprocessing.StandardScaler()
    sc.fit(rssi_all)
    standarded_rssi = sc.transform(rssi_all)
    
    def count_data_num(d, n):
        sum = 0
        for i in range(n):
            sum += len(d[i])
        return sum

    def link_list(li):
        linked_list = []
        for l in li:
            linked_list += l
        return linked_list

    avged_rssis_num = count_data_num(avged_rssis, tester_num)
    rssis_train = standarded_rssi[:avged_rssis_num]
    train_label = link_list(rssi_classes[:tester_num])
    rssis_test = standarded_rssi[avged_rssis_num:]
    test_label = []
    for cls in rssi_classes[tester_num:]:
        test_label += cls
        
    return rssis_train, train_label, rssis_test, test_label

if __name__ == '__main__':
    main()
