import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class graph:
    def __init__(self, save_dir, data, name):

        self.data = data
        self.name = name
        self.save_dir = save_dir

    def monthly_plot(self):
        real = "spx_index"
        pred1 = "pred"
        self.data.index = pd.DatetimeIndex(self.data.index)

        date_s = pd.date_range(
            start=self.data.index[0], end=self.data.index[-1], freq="BMS"
        )
        date_e = pd.date_range(
            start=self.data.index[0], end=self.data.index[-1], freq="BM"
        )

        for num, st in enumerate(range(0, len(date_s), 15)):

            min = self.data.values.min()
            max = self.data.values.max()

            # epp = ep(np.array(data[real]), np.array(data[pred1]))
            # ept = ep(np.array(data[real]), np.array(data[pred2]))
            date_index = []
            mse_ANS = []
            mmse_ANS = []

            hit_ANS = []
            mhit_ANS = []
            stv_ANS = []

            # fig, axs =plt.subplots(nrows=3, ncols=5, figsize=(4,6))

            fig = plt.figure(figsize=(27, 15))
            fig.subplots_adjust(
                left=0.015, right=0.995, bottom=0.025, top=0.95, hspace=0.3, wspace=0.1
            )

            ed = st + 15
            if ed > len(date_e):
                ed = len(date_e)
            for i in range(st, ed):  # len(date_s)):
                df1 = self.data.loc[date_s[i] : date_e[i]]
                df1.index = [str(l) for l in range(1, len(df1) + 1)]

                color_list = ["k", "r", "b"]

                if st < 14:
                    k = i
                else:
                    k = i - st
                ax1 = fig.add_subplot(3, 5, k + 1)
                ax2 = ax1.twinx()

                df1.plot(
                    kind="line",
                    marker=".",
                    ax=ax1,
                    mark_right=False,
                    color=color_list[0:3],
                )
                # k3.plot(kind='bar', ax=ax2, width=0.4, color=color_list[1:3])
                # df9['ANS_ht'].plot(kind='bar', ax=ax2, width=0.2,color='red')
                # df9['CNN_ht'].plot(kind='bar', ax=ax2, width=0.2,color='black')

                smin = df1.values.min()
                smax = df1.values.max()
                # print(smin, smax)

                ax1.set_ylim(smin - (smin * 0.01), smax + 2)
                ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax2.set_ylim(0, 8)

                ax1.legend(
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.7, -0.03),
                    ncol=3,
                    fontsize=9,
                )
                ax2.legend(
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.2, -0.03),
                    ncol=2,
                    fontsize=9,
                )

                ax2.set_yticks([])
                # plt.xticks([])
                ax2.set_title(str(date_s[i])[:7], fontsize=20)
                # plt.title(str(date_s[i])[:7], loc='right')
                # plt.table(cellText=df8.values.round(2), rowLabels=df8.index, colLabels=df8.columns, colWidths=[0.15, 0.15, .15, .15], loc='top', rowLoc='center', cellLoc='center').set_fontsize(12)

                save_path_fig = self.save_dir  # + 'GRAPH\\'
                os.makedirs(save_path_fig, exist_ok=True)
                plt.savefig(save_path_fig + "graph_{}_{}.png".format(self.name, num))
                date_index.append(str(date_s[i])[:7])
