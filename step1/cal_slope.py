import pandas as pd
import numpy as np
import sys
import os
import time
from multiprocessing import Pool

""" Short, Mid, Long term --> 추세 기울기 """


class big3_trend:
    def __init__(self, spx_index, n_info):

        # self.rank = rank_data  #.loc[pd.date_range(start=rank_data.index[0], end=rank_data.index[-1], freq="BM").strftime("%Y-%m-%d")]
        # self.rank.index = pd.DatetimeIndex(self.rank.index)
        self.start, self.end = "2007-01-01", "2018-06-30"
        self.data = spx_index
        self.data.index = pd.DatetimeIndex(self.data.index)

        self.m_range = 250
        self.t_range = 23

        self.n_st = n_info["n_stv"]
        self.n_gap = n_info["n_gap"]
        self.n_size = n_info["n_size"]
        self.n_set = n_info["n_set"]
        self.sv_dir = "../data" + "/{}gap_{}size".format(self.n_gap, self.n_size)
        os.makedirs(self.sv_dir, exist_ok=True)
        self.sv_dir1 = self.sv_dir + "/daily_slope"
        os.makedirs(self.sv_dir1, exist_ok=True)

    def real_N(self, n):
        n_day = self.n_st + (n - 1) * self.n_gap
        return n_day
        # if n_day > 500: return 500
        # else: return n_day

    def tan_slope(self, t1, t2, x_len):

        y1 = self.data.loc[t1].values
        y2 = self.data.loc[t2].values

        return (y1 - y2) / x_len

    def cal_slope(self, q, fix_N=None, how="gm_slope", toward="bw"):  # bw, fw, fw23

        if "fw" in toward:
            if fix_N is not None:

                N = self.real_N(fix_N)
                fw_data = pd.DataFrame()
                fw_mean = pd.DataFrame()

                for i in pd.date_range(start=self.start, end=self.end, freq="B"):

                    if how == "tan_slope":
                        if toward == "fw":
                            f_data = [
                                self.tan_slope(
                                    t, pd.date_range(end=t, freq="B", periods=N)[0], N
                                )
                                - 1
                                for t in pd.date_range(
                                    start=i, freq="B", periods=self.t_range
                                )
                            ]
                        elif toward == "fw23":
                            f_data = self.tan_slope(
                                i,
                                pd.date_range(start=i, freq="B", periods=self.t_range)[
                                    -1
                                ],
                                self.t_range,
                            )

                    elif how == "gm_slope":
                        if toward == "fw":
                            f_data = [
                                (
                                    self.data.loc[t].values
                                    / self.data.loc[
                                        pd.date_range(end=t, freq="B", periods=N)[0]
                                    ].values
                                )
                                ** (252 / N)
                                - 1
                                for t in pd.date_range(
                                    start=i, freq="B", periods=self.t_range
                                )
                            ]
                        elif toward == "fw23":
                            f_data = self.tan_slope(
                                i,
                                pd.date_range(start=i, freq="B", periods=self.t_range)[
                                    -1
                                ],
                                self.t_range,
                            )
                    # fw_23 = pd.concat([fw_23, pd.DataFrame({'fw_23': t_data}, index=[i])])
                    try:
                        fw_data = pd.concat(
                            [
                                fw_data,
                                pd.DataFrame(
                                    {"f_{}".format(t): v for t, v in enumerate(f_data)},
                                    index=[i],
                                ),
                            ]
                        )
                        fw_mean = pd.concat(
                            [
                                fw_mean,
                                pd.DataFrame(
                                    {"{}".format(fix_N): np.mean(f_data)}, index=[i]
                                ),
                            ]
                        )
                    except:
                        fw_data = pd.concat(
                            [fw_data, pd.DataFrame({"fw_23": f_data}, index=[i])]
                        )

                fw_data.to_csv(
                    self.sv_dir1 + "/n{}_{}_total_{}.csv".format(fix_N, how, toward)
                )
                try:
                    fw_mean.to_csv(
                        self.sv_dir + "/n{}_{}_{}_mean.csv".format(fix_N, how, toward)
                    )
                except:
                    pass

            else:
                n_total = pd.DataFrame()
                for n in self.n_set[q : q + int(self.n_size / 3)]:

                    N = self.real_N(n)
                    fw_data = pd.DataFrame()
                    fw_mean = pd.DataFrame()
                    for i in pd.date_range(start=self.start, end=self.end, freq="B"):

                        if how == "tan_slope":
                            if toward == "fw":
                                f_data = [
                                    self.tan_slope(
                                        t,
                                        pd.date_range(end=t, freq="B", periods=N)[0],
                                        N,
                                    )
                                    - 1
                                    for t in pd.date_range(
                                        start=i, freq="B", periods=self.t_range
                                    )
                                ]
                            elif toward == "fw23":
                                f_data = self.tan_slope(
                                    i,
                                    pd.date_range(
                                        start=i, freq="B", periods=self.t_range
                                    )[-1],
                                    self.t_range,
                                )

                        elif how == "gm_slope":
                            if toward == "fw":
                                f_data = [
                                    (
                                        self.data.loc[t].values
                                        / self.data.loc[
                                            pd.date_range(end=t, freq="B", periods=N)[0]
                                        ].values
                                    )
                                    ** (252 / N)
                                    - 1
                                    for t in pd.date_range(
                                        start=i, freq="B", periods=self.t_range
                                    )
                                ]
                            elif toward == "fw23":
                                f_data = (
                                    self.data.loc[
                                        pd.date_range(
                                            start=i, freq="B", periods=self.t_range
                                        )[-1]
                                    ].values
                                    / self.data.loc[i].values
                                ) ** (252 / self.t_range) - 1

                        # fw_23 = pd.concat([fw_23, pd.DataFrame({'fw_23': t_data}, index=[i])])
                        try:
                            fw_data = pd.concat(
                                [
                                    fw_data,
                                    pd.DataFrame(
                                        {
                                            "f_{}".format(t): v
                                            for t, v in enumerate(f_data)
                                        },
                                        index=[i],
                                    ),
                                ]
                            )
                            fw_mean = pd.concat(
                                [
                                    fw_mean,
                                    pd.DataFrame(
                                        {"{}".format(n): np.mean(f_data)}, index=[i]
                                    ),
                                ]
                            )
                        except:
                            fw_data = pd.concat(
                                [fw_data, pd.DataFrame({"fw_23": f_data}, index=[i])]
                            )

                    fw_data.to_csv(
                        self.sv_dir1 + "/n{}_{}_total_{}.csv".format(n, how, toward)
                    )
                    try:
                        n_total = pd.concat([n_total, fw_mean], axis=1)
                        n_total.to_csv(
                            self.sv_dir
                            + "/{}_total_{}_mean_{}clust.csv".format(how, toward, q)
                        )
                    except:
                        pass

        elif toward == "bw":
            if fix_N is not None:

                N = self.real_N(fix_N)
                hist_slope = pd.DataFrame()
                bw_mean = pd.DataFrame()

                for i in pd.date_range(start=self.start, end=self.end, freq="B"):

                    if how == "tan_slope":
                        m_data = [
                            self.tan_slope(
                                t, pd.date_range(end=t, freq="B", periods=N)[0], N
                            )
                            for t in pd.date_range(
                                end=i, freq="B", periods=self.m_range
                            )
                        ]

                    elif how == "gm_slope":
                        m_data = [
                            (
                                self.data.loc[t].values
                                / self.data.loc[
                                    pd.date_range(end=t, freq="B", periods=N)[0]
                                ].values
                            )
                            ** (252 / N)
                            - 1
                            for t in pd.date_range(
                                end=i, freq="B", periods=self.m_range
                            )
                        ]
                    hist_slope = pd.concat(
                        [
                            hist_slope,
                            pd.DataFrame(
                                {"b_{}".format(t + 1): v for t, v in enumerate(m_data)},
                                index=[i],
                            ),
                        ]
                    )
                    bw_mean = pd.concat(
                        [
                            bw_mean,
                            pd.DataFrame({"bw_250_mean": np.mean(m_data)}, index=[i]),
                        ]
                    )

                hist_slope.to_csv(self.sv_dir1 + "/n_{}_bw_slope.csv".format(fix_N))
                bw_mean.to_csv(self.sv_dir + "/n_{}_bw_mean.csv".format(fix_N))

            else:
                n_total = pd.DataFrame()
                for n in self.n_set[q : q + int(self.n_size / 3)]:
                    print(n)
                    N = self.real_N(n)
                    hist_slope = pd.DataFrame()
                    bw_mean = pd.DataFrame()
                    for i in pd.date_range(start=self.start, end=self.end, freq="B"):

                        if how == "tan_slope":
                            m_data = [
                                self.tan_slope(
                                    t, pd.date_range(end=t, freq="B", periods=N)[0], N
                                )
                                for t in pd.date_range(
                                    end=i, freq="B", periods=self.m_range
                                )
                            ]

                        elif how == "gm_slope":
                            m_data = [
                                (
                                    self.data.loc[t].values
                                    / self.data.loc[
                                        pd.date_range(end=t, freq="B", periods=N)[0]
                                    ].values
                                )
                                ** (252 / N)
                                - 1
                                for t in pd.date_range(
                                    end=i, freq="B", periods=self.m_range
                                )
                            ]

                        bw_mean = pd.concat(
                            [
                                bw_mean,
                                pd.DataFrame(
                                    {"n_{}".format(n): np.mean(m_data)}, index=[i]
                                ),
                            ]
                        )
                        hist_slope = pd.concat(
                            [
                                hist_slope,
                                pd.DataFrame(
                                    {
                                        "b_{}".format(t + 1): v
                                        for t, v in enumerate(m_data)
                                    },
                                    index=[i],
                                ),
                            ]
                        )
                    print(bw_mean)
                    n_total = pd.concat([n_total, bw_mean], axis=1)
                    hist_slope.to_csv(self.sv_dir1 + "/n_{}_bw_slope.csv".format(n))
                    n_total.to_csv(
                        self.sv_dir + "/{}_total_bw_mean_{}clust.csv".format(how, q)
                    )
                    print(n_total.head())
        # while len([i for i in os.listdir(self.sv_dir) if 'total_{}_mean'.format(toward) in i]) != 3:
        #     time.sleep(10*60)

    def total_slope(self, how, toward):

        new = pd.DataFrame()
        for i in os.listdir(self.sv_dir):
            if "total_{}_mean".format(toward) in i:
                new = pd.concat(
                    [new, pd.read_csv(os.path.join(self.sv_dir, i), index_col=0)],
                    axis=1,
                )

        total = pd.DataFrame()
        for n in self.n_set:
            total = pd.concat([total, new["n_{}".format(n)]], axis=1)
        print(total)
        total.to_csv(
            self.sv_dir
            + "/{}_total_{}_mean_{}clust.csv".format(how, toward, self.n_size)
        )
        term_avg = self.group_avg_slope(total)
        term_avg.to_csv(
            self.sv_dir + "/{}_group_avg_slope_{}clust.csv".format(how, self.n_size)
        )
        top_n = self.match_term_top_N(total, term_avg)
        top_n.to_csv(self.sv_dir + "/each_term_match_top_N.csv")

    ## 단/중/장 각각 평균기울기
    def group_avg_slope(self, slope_data):

        term_avg = pd.DataFrame()
        cls = int(len(slope_data.columns) / 3)

        for num, i in enumerate(range(0, len(slope_data.columns), cls)):
            print(slope_data.columns[i : i + cls])
            avg_slope = slope_data[slope_data.columns[i : i + cls]].mean(axis=1)
            if num == 0:
                col = ["S"]
            elif num == 1:
                col = ["M"]
            else:
                col = ["L"]
            term_avg = pd.concat(
                [
                    term_avg,
                    pd.DataFrame(avg_slope.values, index=avg_slope.index, columns=col),
                ],
                axis=1,
            )
        return term_avg

    ## 단/중/장 평균기울기 가장 유사한 topN( MAE 기준 )
    def match_term_top_N(self, n_slope, group_avg_slope):

        daily = group_avg_slope.index
        total = pd.DataFrame()
        for term in group_avg_slope.columns:

            mae = pd.DataFrame()
            top4 = pd.DataFrame()
            for n in n_slope.columns:
                mae = pd.concat(
                    [
                        mae,
                        pd.DataFrame(
                            {n: abs(n_slope[n].values - group_avg_slope[term].values)},
                            index=daily,
                        ),
                    ],
                    axis=1,
                )

            for t in daily:
                top4 = pd.concat(
                    [
                        top4,
                        pd.DataFrame(
                            {
                                "{}_{}".format(term, i): int(v.split("_")[1])
                                for i, v in enumerate(
                                    mae.loc[t].sort_values(ascending=True)[:4].index
                                )
                            },
                            index=[t],
                        ),
                    ]
                )
            total = pd.concat([total, top4], axis=1)
        return total


def concat(n1, n2, n_set):
    new = pd.DataFrame()
    concat = pd.concat([n1, n2], axis=1)
    for i in n_set:
        new = pd.concat([new, pd.DataFrame(concat["n_{}".format(i)])], axis=1)
    return new
