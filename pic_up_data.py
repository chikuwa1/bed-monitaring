import pandas as pd


# CSVファイルを読み込む
file_name = "/home/chiaki/bed-monitoring/csv/log1208/zero.csv"
raw_bed_data = pd.read_csv(file_name, header=None)

# 時刻・タグ名・RSSI値を取り出す
time_idx, tag_idx, rssi_idx = 0, 8, 11
raw_bed_data = raw_bed_data[[time_idx, tag_idx, rssi_idx]]

# 時刻を秒数に変換する
raw_bed_data['time'] = (raw_bed_data['time'] -
                        raw_bed_data['time'].min()).dt.total_seconds()  # 最初の時刻を0とする

# タグごとに時刻とRSSI強度をまとめる
bed_data = {}
for tag, data in raw_bed_data.groupby('tag'):
    bed_data[tag] = data[['time', 'rssi']]

print(bed_data)
