from pprint import pprint
import numpy as np
import pandas as pd
from sklearn import preprocessing


TIME_IDX = 1
TAG_IDX = 8
RSSI_IDX = 10

# この3つはリテラルが上のように決まっている


class Reshaper:
    """ベッドデータの機械学習のためのデータ整形クラス．

    このクラスの目的は，ベッドデータの機械学習の精度向上のためある一定のデータ区間の
    RSSI値の平均値を取得してそのデータに対応する姿勢クラスの両方を得ることである．

    Attributes:
        data_num_perblock (int)      : RSSI値の平均を取るデータ数 = 50
        no_reaction_rssi  (float)    : センサー無反応の時のRSSIの代替値 =-105.0
        csv_path          (str)      : CSVファイルのパス = ./csv/log1208/
        tester_name       (list)     : 被験者の名前 -> tester_name.txtに書いてる
        class_num         (int)      : 姿勢クラス数 = 7
        tag_name          (set[str]) : センサータグの名前 = E280116060000204AC6Axxxx(xが違う)
    """

    def __init__(self, data_num_perblock, no_reaction_rssi, csv_path,
                 tester_name, class_num):
        self.data_num_perblock = data_num_perblock # RSSI値の平均を取るデータ数 = 50
        self.no_reaction_rssi = no_reaction_rssi # センサー無反応の時のRSSIの代替値 =-105.0
        self.csv_path = csv_path # CSVファイルのパス = ./csv/log1208/
        self.tester_name = tester_name # tester_name.txtに書いてる
        self.class_num = class_num # 姿勢クラス数 = 7
        self.tag_num = 6

    def get_learnable_multi_train_data(self, train_count):
        """複数人の CSV データから平均化した RSSI 値とそれに対応する姿勢クラスを得る

        Returns:
            train_rssis (list): 訓練用 RSSI  ->>今回はこいつが何か調べるぞ
            train_label (list): 訓練用ラベル
            test_rssis  (list): テスト用 RSSI
            test_label  (list): テスト用ラベル
        """
        bed_data = self.__import_csv()
        rssis = self.__extract_rssis(bed_data)
        avged_rssis, posture_classes = self.__take_block_rssi_avg(rssis)
        standarded_rssis = self.__standardize(avged_rssis)
        train_rssis, train_label, test_rssis, test_label = self.__divide_data_units_tester(avged_rssis,
                                                                                           standarded_rssis,
                                                                                           posture_classes,
                                                                                           train_count)

        return train_rssis, train_label, test_rssis, test_label

    def get_learnable_single_train_data(self, train_rate):
        """ 1 人分の CSV データから平均化した RSSI 値とそれに対応する姿勢クラスを得る

        Returns:
            train_rssis (list): 訓練用 RSSI
            train_label (list): 訓練用ラベル
            test_rssis  (list): テスト用 RSSI
            test_label  (list): テスト用ラベル
        """
        bed_data = self.__import_csv()
        rssis = self.__extract_rssis(bed_data)
        avged_rssis, posture_classes = self.__take_block_rssi_avg(rssis)

        standarded_rssis = [self.__standardize(rssis) for rssis in avged_rssis]

        train_rssis, train_label, test_rssis, test_label = [], [], [], []
        for i, stded_rssis in enumerate(standarded_rssis):
            train_r, train_l, test_r, test_l = self.__divide_data_train_rate(stded_rssis,
                                                                             posture_classes[i],
                                                                             train_rate)
            train_rssis.append(train_r)
            train_label.append(train_l)
            test_rssis.append(test_r)
            test_label.append(test_l)

        return train_rssis, train_label, test_rssis, test_label

    def __import_csv(self):
        """ CSV ファイルを読み込む

        Returns:
            list: センサーのRSSI値のデータ
                  bed_data[<被験者idx(姿勢クラス0は被験者なし)>]
                          [<姿勢クラス>]
                          [<"time"or"tag"or"rssi">]
                          [<dataidx(idx>=1よりデータ部分)>]
        """

        tester_num = len(self.tester_name) # tester_numは被験者の人数=11
        bed_data = [[] for _ in range(tester_num)] # bed_data[11][?]
        file_name = f"{self.csv_path}zero.csv" # ベッド上に誰もいないときのcsv
        # CSVファイルから時間，タグ名，RSSI値を得る
        raw_bed_data = pd.read_csv(file_name, header=None)[[TIME_IDX, TAG_IDX, RSSI_IDX]]
        # pandas.DataFrameは二次元の表形式のデータ
        # header=Noneでヘッダーなし(列名(上の方の名前)が表示されない)
        # raw_bed_data はfile_nameのcsvから時間とタグ名とRSSI値が入る->[[TIME_IDX, TAG_IDX, RSSI_IDX]]は多くのインデックスの中からこれらを厳選するために記述必要
        zero_cls_data = []
        element_num = len(raw_bed_data) // tester_num # CSVファイルからとったデータを11で割る(あまり切り捨て) element_num=2406?? 割る目的わからん
        # もしかしたらベット上に誰もいない状態を11で割って，一人どれくらいずつのRSSI値を割り振るのかを表してる??

        for i in range(1, len(raw_bed_data), element_num):
            data = raw_bed_data[i:(i+element_num)] # dataにraw_bed_dataの11等分したものを順に入れている（dataは表データ）
            zero_cls_data.append(data.reset_index()) 
            # reset_index()で0から連番で設定する＋indexというカラムが追加される（つまりこの時にindex,time,tag,rssiという形のlistになる）
            # listであるzero_cls_dataにはraw_bed_dataのインデックスが[1~,element+1~,...,len(raw_bed_data)-1-ele,emt_num～]の11個のcsvのデータが入っているってこと？

        for i in range(len(zero_cls_data)): # len(zero_cls_data)=11？
            zero_cls_data[i].columns = [u"index", u"time", u"tag", u"rssi"] # 列名(カラム名)の指定 index入れたらindexついてくるのか？
            # u…Unicodeに変換する

        for i, c_0_d in enumerate(zero_cls_data):
            bed_data[i].append(c_0_d) # bed_dataのiごとにzero_cls_dataのindex,time,tag,rssiを入れてる->iとindexは同じ数値なはず

        # bed_dataにはindex,time,tag,rssi（indexは0～10まで）が入っている

        for i, tester in enumerate(self.tester_name): # 連番と被験者名をi,testerに入れる
            for cls_num in range(1, self.class_num): # class_num は姿勢クラス数=7->　1～6までをcls_numに入れて繰り返す
                file_name = f"{self.csv_path}{tester}_{str(cls_num)}.csv" # 被験者名_姿勢クラス.csvというすべてのファイルを見る
                # CSVファイルから時間，タグ名，RSSI値を得る
                raw_bed_data = pd.read_csv(file_name, header=None)[[TIME_IDX, TAG_IDX, RSSI_IDX]] # 上記同様，時間，タグ，RSSI値をraw_bed_dataに入れる
                raw_bed_data.columns = [u"time", u"tag", u"rssi"] # これはインデックス名がない！！！なぜ故．→多分reset_indexしてないから
                bed_data[i].append(raw_bed_data) # appendは追加の動き(上書きじゃないよ)
                # イメージとしては[[zoroでのindex,time,tag,rssi], [(appendされた姿勢1の)time,tag,rssi], [(appendされた姿勢2の)time,tag,rssi],..., [(appendされた姿勢6の)time,tag,rssi]]

        return bed_data # bed_data[i]にはtester i の姿勢クラス0~6ごとにtime,tag,rssiデータが入っていることになる(iは0～10)

    def __extract_rssis(self, bed_data):
        """ CSV ファイルから時間・タグ名・RSSI 値を抽出する
        Args:
            bed_data (list): センサーのRSSI値のデータ

        Returns:
            list: 同時間に取得された各タグの RSSI 値
                  無反応のタグの RSSI は no_reaction_rssi とする
        """

        tester_num = len(self.tester_name)
        # RSSI 値のリスト
        rssis = [[[] for _ in range(self.class_num)]
                 for _ in range(tester_num)]

        # タグID
        tag_name = ("E280116060000204AC6AD0EC",
                    "E280116060000204AC6AD0E6",
                    "E280116060000204AC6AD1FE",
                    "E280116060000204AC6AD1FD",
                    "E280116060000204AC6AC8F0",
                    "E280116060000204AC6AD1FC")
        # tag_nameの逆引き辞書
        tag_name_dict = {tag_name[0]: 0,
                         tag_name[1]: 1,
                         tag_name[2]: 2,
                         tag_name[3]: 3,
                         tag_name[4]: 4,
                         tag_name[5]: 5}

        def init_rssi():
            return [self.no_reaction_rssi for _ in range(self.tag_num)]

        for tester_num, d in enumerate(bed_data):
            for cls_num, data in enumerate(d):
                time = data["time"][1]
                rssi = init_rssi()
                for i in range(1, len(data)):
                    if time != data["time"][i]:
                        rssis[tester_num][cls_num].append(rssi)
                        rssi = init_rssi()
                    tag = data["tag"][i]
                    sensor_idx = tag_name_dict[tag]
                    rssi[sensor_idx] = float(data["rssi"][i])
                rssis[tester_num][cls_num].append(rssi)

        # pprint(rssis[<被験者番号>][<姿勢クラス>])
        # print(len(rssis[0][2]))
        return rssis

    def __take_block_rssi_avg(self, rssis):
        """ 各タグの同時間に取得された RSSI 値を平均化する
        Args:
            rssis (list): 同時間に取得された各タグの RSSI 値

        Returns:
            list: 同時間に取得された各タグの RSSI 値の平均
        """

        tester_num = len(self.tester_name)
        avged_rssis = [[] for _ in range(tester_num)]
        posture_classes = [[] for _ in range(tester_num)]

        def avg(li):
            return sum(li) / len(li)

        for tester, d in enumerate(rssis):
            for cls_num, data in enumerate(d):
                # 先頭の平均値データの作成
                rssi_block = [[data[i][j] for i in range(self.data_num_perblock)]
                              for j in range(self.tag_num)]
                avged_rssi = [avg(rssi_block[i]) for i in range(self.tag_num)]
                avged_rssis[tester].append(avged_rssi)
                posture_classes[tester].append(cls_num)

                # それ以降の平均値データの作成
                for rssi in data[self.data_num_perblock:]:
                    avged_rssi = []
                    for i in range(self.tag_num):
                        rssi_block[i].pop(0)
                        rssi_block[i].append(rssi[i])
                        rssi_avg = avg(rssi_block[i])
                        avged_rssi.append(rssi_avg)
                    avged_rssis[tester].append(avged_rssi)
                    posture_classes[tester].append(cls_num)

        return avged_rssis, posture_classes

    def __standardize(self, rssis):
        """ 各タグの同時間に取得された RSSI 値を標準化する

        Args:
            rssis (list): 同時間に取得された各タグの RSSI 値

        Returns:
            list: 同時間に取得された各タグの RSSI 値の標準化
        """
        if len(rssis) == len(self.tester_name):
            all_rssis = []
            for rssi in rssis:
                all_rssis += rssi
        else:
            all_rssis = rssis

        sc = preprocessing.StandardScaler()
        sc.fit(all_rssis)
        standarded_rssi = sc.transform(all_rssis)

        # print(standarded_rssi[0])

        return standarded_rssi

    def __divide_data_units_tester(self, avged_rssis, standarded_rssis, posture_classes, train_count):
        """ 学習データとテストデータに分割する

        Args:
            avged_rssis (list): 同時間に取得された各タグの RSSI 値の平均
            standarded_rssis (list): 同時間に取得された各タグの RSSI 値の標準化
            posture_classes (list): 同時間に取得された各タグの RSSI 値の姿勢クラス
            train_count (int): 学習データの割合

        Returns:
            list: 学習データ
        """
        avged_rssis_num = 0
        for rssi in avged_rssis[:train_count]:
            avged_rssis_num += len(rssi)

        train_rssis = standarded_rssis[:avged_rssis_num]

        train_label = []
        for rssi in posture_classes[:train_count]:
            train_label += rssi

        test_rssis = standarded_rssis[avged_rssis_num:]

        test_label = []
        for cls in posture_classes[train_count:]:
            test_label += cls

        return train_rssis, train_label, test_rssis, test_label

    def __divide_data_train_rate(self, rssis, posture_classes, train_rate):
        cls_start_idx = 0
        prev_cls = 0    # 1回前のループでの姿勢クラス
        train_rssis, train_label, test_rssis, test_label = None, [], None, []
        for i, cls in enumerate(posture_classes):
            if prev_cls == (self.class_num - 1):
                train_start = cls_start_idx
                train_end = cls_start_idx + int((len(rssis) - cls_start_idx) * train_rate)
                test_start = train_end
                test_end = len(rssis)
                
                train_rssis = np.concatenate([train_rssis, rssis[train_start:train_end]], axis=0)
                test_rssis = np.concatenate([test_rssis, rssis[test_start:test_end]], axis=0)
                train_label += posture_classes[train_start:train_end]
                test_label += posture_classes[test_start:test_end]
                break
            if prev_cls != cls:
                train_start = cls_start_idx
                train_end = cls_start_idx + int((i - cls_start_idx) * train_rate)
                test_start = train_end
                test_end = i
                if prev_cls == 0:
                    train_rssis = rssis[train_start:train_end]
                    test_rssis = rssis[test_start:test_end]
                else:
                    train_rssis = np.concatenate([train_rssis, rssis[train_start:train_end]], axis=0)
                    test_rssis = np.concatenate([test_rssis, rssis[test_start:test_end]], axis=0)
                train_label += posture_classes[train_start:train_end]
                test_label += posture_classes[test_start:test_end]
                cls_start_idx = i
                prev_cls += 1

        return train_rssis, train_label, test_rssis, test_label
    
    def get_rssis(self):
        bed_data = self.__import_csv()
        rssis = self.__extract_rssis(bed_data)
        return rssis