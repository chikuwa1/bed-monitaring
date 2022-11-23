# プログラムを理解するためにメモする用のダミープログラム

import sys
import yaml # YAMLファイルを読み書きするのに必要

from reshaper import Reshaper
from sklearn.linear_model import SGDClassifier 
# SGDClassifierは確率的勾配降下(SGD)を行う
# SGDはランダムなデータを抽出して関数の最小値を見つける方法
# ディープラーニングでは誤差が最小値となる重み(パラメータ)を探すために使われる->何に使う？？

from sklearn.metrics import accuracy_score, confusion_matrix 

# confusion_matrixで混合行列を作っている
# 混合行列…正事例と負事例の予測の結果をまとめた表(例:犬が犬だと識別されたTP,犬が犬でないと識別されたFN,犬出ないのが犬と識別されたFP,犬でないのが犬出ないと識別されたTN)
# 機械学習を用いたクラス分類の精度を評価するには，混合行列を用いて正しい識別ト誤った識別の件数を比較することが一般的
# 混合行列を用いれば，正解率，精度，検出率，F値がわかる
# accuracy_scoreは正解率を計算するメソッド
# 正解率:本来ポジティブに分類すべきアイテムをポジティブに分類し、本来ネガティブに分類すべきアイテムをネガティブに分類できた割合

from sklearn.neighbors import KNeighborsClassifier
# KNeighborsClassifierはK-近傍法(K-NN)
# K-NNは特徴空間上において近くにあるK個オブジェクトのうち最も数が多いクラスに分類する
# k値によって精度が変化->最適なk値を設定する必要あり

from sklearn.svm import LinearSVC
# LinearSVCはSVMのひとつ
# SVMは教師あり学習のクラス分類と、回帰のできる機械学習アルゴリズム
# 線形(超平面)によって分類する
# LinearSVCは各サンプルからの距離が最大になるように境界線を求める手法

# main関数
def main():
    # with open("ファイル名") as f: ("ファイル名"のファイルをオープンした状態をfとする)
    # withで作成したファイルオブジェクトを自動的に解放するようにできる
    # -> with内が終わればクローズされるし，解放もされるってこと

    with open('tester_name.txt', 'r') as f: #tester_name.txtは読み込み専用でopen
        testers = f.read().splitlines()  ## 改行文字を消去
        # tester_name.txtには11名の名前が改行され記述されている
        # testersは11名の名前が入っているリスト
        # read()は読み込む->testersにfの内容を読み込む
        # splitlines()によって改行ごとにリストに挿入（改行文字は削除される）

    with open('config.yml') as yml: #config.ymlに色々書いてるから確認すべし
        config = yaml.safe_load(yml)
        # configにymlファイルを読み込む

    csv_path = config['csv_path'] # CSVファイルのパス= ./csv/log1208/
    data_range = config['data_range'] # データ整形時に平均値を取るデータの範囲=50
    knn_neighbors = config['knn_neighbors'] # K近傍法の近傍オブジェクト数=3
    no_reaction_rssi = config['no_reaction_rssi'] # 無反応のセンサーに対する代替のRSSI値=-105.0
    posture_class_num = config['posture_class_num'] # 姿勢クラス数=7
    err_message = f'''usage: python clustering.py <clustering method> <count of train(int:1~{len(testers)-1})>
    available clustering method:
        svc     Linear SVC(SVM Classification)
        sgd     SGD(Stochastic Gradient Descent)
        kneigh  K-NeighborsClassifier (K={knn_neighbors})
    '''
    # err_message
    # 利用方法：python clustering.py の <クラスタリング方法> <訓練数(整数:1~testersの数-1まで)>
    # 利用するクラスタリング方法: SVC SGD Knrigh


    ## コマンドライン引数のバリデーション（コマンドライン引数が正しいかどうかの確認）
    # コマンドラインは　python ○○.py <クラスタリングの手法> <訓練数> の形にする
    if len(sys.argv) != 3:
        error(err_message)
    # もしコマンドラインの長さが3じゃなかったらエラー-> $ python ○○.py は長さ1

    method = sys.argv[1] # コマンドライン引数の2つめに方法を打ち込み，それをmethodに代入

    if method != 'svc' and method != 'sgd' and method != 'kneigh': # もしどの方法でもなかったらエラー
        error(err_message)

    try:
        train_count = int(sys.argv[2]) # コマンドライン引数の3つめを整数にした値をtrain_countに代入->argv[2]には訓練数を入れる
    except ValueError: # 値がおかしかったらエラー>intで整数にならないとか
        error(err_message)
    
    if not 1 <= train_count <= (len(testers) - 1): # 訓練数が1以上tester-1以下でないならエラー
        error(err_message)

    with open('tester_name.txt', 'r') as f: # 38行に同じコードがある．．．
        tester_name = f.read().splitlines() # 改行文字を消去

    reshaper = Reshaper(data_range, no_reaction_rssi,
                        csv_path, tester_name, posture_class_num)
    # data_range=50, no_reaction_rssi=-105.0, csv_path=./csv/log1208/, posture_class_num=7 (48行目付近)
    # tester_name は11名の名前が入ったリスト(39,83行目)
    # これらをReshaperへ引き渡す

    rssis_train, train_label, rssis_test, test_label = reshaper.get_learnable_multi_train_data(train_count)
    # Reshaperのget_learnable_multi_train_data関数を用いて訓練用，テスト用のRSSIとラベルを代入
    
    # !!ここでのラベルは何のラベル？？->
    
    ## 手法の選択(コマンドライン引数によって決定)
        # もしクラスタリング方法がSVCだったら
    if method == 'svc':
        ## ConvergenceWarningが出現する場合にはmax_iter=10000を追加する
        # ConvergenceWarningは収束エラーで，SVCの時直線で区分できない際に出てくる
        clf_result = LinearSVC(loss='hinge', random_state=0) # loss=ヒンジ損失, 乱数はrandom_stateより同じでLinearSVCを行い結果をclf_resultへ
        
        # LinearSVC のパラメータ
        # <loss=評価関数で，ヒンジ損失か二乗ヒンジ(デフォルト)>
        # <random_stateはランダム化がアルゴリズムの一部である際に制御するために設定→なし:繰り返すと異なる乱数が生成，整数:繰り返しても同じ乱数が生成(0or42が多い)>

        # もしSGDだったら
    elif method == 'sgd':
        clf_result = SGDClassifier(loss="hinge")

        #もしknn法だったら
    elif method == 'kneigh':
        clf_result = KNeighborsClassifier(n_neighbors=knn_neighbors) # knn_neighbors=3

    # 姿勢クラスの再割り当て(クラス数7→3)
    for i, label in enumerate(train_label): # train_labelは訓練用ラベル
        if label == 1 or label == 2 or label == 3:
            train_label[i] = 1
        if label == 4 or label == 5 or label == 6:
            train_label[i] = 2
    for i, label in enumerate(test_label): # test_labelはテスト用ラベル
        if label == 1 or label == 2 or label == 3:
            test_label[i] = 1
        if label == 4 or label == 5 or label == 6:
            test_label[i] = 2

    # 学習
    clf_result.fit(rssis_train, train_label)
    
    # 予測
    pre = clf_result.predict(rssis_test)

    # Confusion Matrix出力
    print('---------- Confusion Matrix ----------')
    print(confusion_matrix(test_label, pre))
    print('--------------------------------------')

    # 正答率計算
    ac_score = accuracy_score(test_label, pre)
    print("正答率 =", ac_score)


def error(err_str):
    print(err_str)
    exit()


if __name__ == '__main__':
    main()
