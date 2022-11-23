from itertools import combinations
import matplotlib.pyplot as plt


class ResultDrawer:
    def __init__(self, train_rssis, train_label, posture_class_num=6):
        self.train_rssis = train_rssis
        self.train_label = train_label
        self.posture_class_num = posture_class_num

    def plot_by_posture(self):
        moke = [[] for _ in range(self.posture_class_num)]
        for data in self.train_rssis:
            # print(data)
            for i, d in enumerate(data):
                moke[i].append(d)

        comb = list(combinations([i for i in range(6)], 2))
        for c in comb:
            plt.clf()
            plt.close()
            plt.xlabel(str(c[0]), fontsize=18, loc="right")
            plt.ylabel(str(c[1]), fontsize=18, loc="top")
            print(len(self.train_label))
            plt.scatter(moke[c[0]], moke[c[1]], c=self.train_label, s=10, cmap=plt.cm.coolwarm)
            plt.savefig(f"/mnt/c/Users/chiaki/Desktop/result-{str(c[0])}-{str(c[1])}.png")

    def plot_by_person(self):
        pass
