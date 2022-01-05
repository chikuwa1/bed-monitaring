import pandas as pd

class Reshaper:
    """ベッドデータの機械学習のためのデータ整形クラス．

    このクラスの目的は，ベッドデータの機械学習の精度向上のためある一定のデータ区間の
    RSSI値の平均値を取得してそのデータに対応するクラスの両方を得ることである．

    Attributes:
    data_num_perblock (int)      : RSSI値の平均を取るデータ数
    no_reaction_rssi  (float)    : センサー無反応の時のRSSIの代替値
    csv_path          (str)      : CSVファイルのパス
    tester_name       (list)     : 被験者の名前
    class_num         (int)      : クラス(分類する姿勢)数
    tag_name          (set[str]) : センサータグの名前
    """

    def __init__(self, data_num_perblock, no_reaction_rssi,
                 csv_path, tester, class_num):
        self.data_num_perblock = data_num_perblock
        self.no_reaction_rssi = no_reaction_rssi
        self.csv_path = csv_path
        self.tester_name = tester
        self.class_num = class_num
        self.tag_name = ('E280116060000204AC6AD0EC',
                         'E280116060000204AC6AD0E6',
                         'E280116060000204AC6AD1FE',
                         'E280116060000204AC6AD1FD',
                         'E280116060000204AC6AC8F0',
                         'E280116060000204AC6AD1FC')

    def parse_avged_rssi_and_cls(self):
        """CSVデータから平均化したRSSI値とそれに対応する姿勢クラスを得る

        Returns:
            list: 平均化したRSSI値
            list: avged_rssisに対応するクラス
        """
        print('File loading...          ', end='')
        bed_data = self.__load_csv()
        print('\033[32m'+'Done'+'\033[0m')

        print('Data reshaping...        ', end='')
        rssis = self.__get_rssis(bed_data)
        avged_rssis, rssi_classes = self.__take_block_rssi_avg(rssis)
        print('\033[32m'+'Done'+'\033[0m')

        return avged_rssis, rssi_classes

    def __load_csv(self):
        """CSVファイルの読み込み，

        Returns:
            list: [description]
        """

        tester_num = len(self.tester_name)
        bed_data = [[] for _ in range(tester_num)]
        file_name = f'{self.csv_path}zero.csv'
        # CSVファイルから時間，タグ名，RSSI値を得る
        raw_bed_data = pd.read_csv(file_name, header=None)[[1, 8, 10]]

        zero_cls_data = []
        element_num = len(raw_bed_data) // tester_num

        for i in range(1, len(raw_bed_data), element_num):
            data = raw_bed_data[i:(i+element_num)]
            zero_cls_data.append(data.reset_index())

        for i in range(len(zero_cls_data)):
            zero_cls_data[i].columns = [u'index', u'time', u'tag', u'rssi']

        for i, c_0_d in enumerate(zero_cls_data):
            bed_data[i].append(c_0_d)

        for i, tester in enumerate(self.tester_name):
            for cls_idx in range(1, self.class_num):
                file_name = f'{self.csv_path}{tester}_{str(cls_idx)}.csv'
                # CSVファイルから時間，タグ名，RSSI値を得る
                raw_bed_data = pd.read_csv(file_name, header=None)[[1, 8, 10]]
                raw_bed_data.columns = [u'time', u'tag', u'rssi']
                bed_data[i].append(raw_bed_data)

        print(type(bed_data[1][0]))
        return bed_data

    def __get_rssis(self, bed_data):
        """RSSI値を取得

        Args:
            bed_data (list): センサーのRSSI値のデータ．
                             bed_data[<被験者idx(クラス0は被験者なし)>]
                                     [<姿勢のクラスidx>]
                                     [<'time'or'tag'or'rssi'>]
                                     [<dataidx(idx>=1よりデータ部分)>]

        Returns:
            list: [description]
        """

        tester_num = len(self.tester_name)
        rssis = [[[] for _ in range(self.class_num)]
                 for _ in range(tester_num)]

        tag_dict = {self.tag_name[0]: 0,
                    self.tag_name[1]: 1,
                    self.tag_name[2]: 2,
                    self.tag_name[3]: 3,
                    self.tag_name[4]: 4,
                    self.tag_name[5]: 5}

        def init_rssi():
            sensor_num = len(self.tag_name)
            return [self.no_reaction_rssi for _ in range(sensor_num)]

        for tester, d in enumerate(bed_data):
            for cls_num, data in enumerate(d):
                time = data['time'][1]
                rssi = init_rssi()
                for i in range(1, len(data)):
                    if time != data['time'][i]:
                        rssis[tester][cls_num].append(rssi)
                        rssi = init_rssi()
                    tag_name = data['tag'][i]
                    sensor_idx = tag_dict[tag_name]
                    rssi[sensor_idx] = float(data['rssi'][i])
                rssis[tester][cls_num].append(rssi)

        return rssis

    def __take_block_rssi_avg(self, rssis):
        """[summary]

        Args:
            rssis ([type]): [description]

        Returns:
            [type]: [description]
        """

        tester_num = len(self.tester_name)
        sensor_num = len(self.tag_name)
        avged_rssis = [[] for _ in range(tester_num)]
        rssi_classes = [[] for _ in range(tester_num)]

        def avg(li):
            return sum(li) / len(li)

        for tester, d in enumerate(rssis):
            for cls_idx, data in enumerate(d):
                # 先頭の平均値データの作成
                rssi_block = [[] for _ in range(sensor_num)]
                avged_rssi = []
                for i in range(sensor_num):
                    for j in range(self.data_num_perblock):
                        rssi_block[i].append(data[j][i])
                    rssi_avg = avg(rssi_block[i])
                    avged_rssi.append(rssi_avg)
                avged_rssis[tester].append(avged_rssi)
                rssi_classes[tester].append(cls_idx)

                # それ以降の平均値データの作成
                for rssi in data[self.data_num_perblock:]:
                    avged_rssi = []
                    for i in range(sensor_num):
                        rssi_block[i].pop(0)
                        rssi_block[i].append(rssi[i])
                        rssi_avg = avg(rssi_block[i])
                        avged_rssi.append(rssi_avg)
                    avged_rssis[tester].append(avged_rssi)
                    rssi_classes[tester].append(cls_idx)

        return avged_rssis, rssi_classes
