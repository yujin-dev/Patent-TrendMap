import pandas as pd
import os
from result_graph import graph
from predict_map import pred_anal


class group_processing:

    """ answer_target 만들기/ 단중장 예측 - slope가장 유사한 N선정(topn개) """

    def __init__(self, data_setting):
        self.total_bw_slope = data_setting["total_bw_slope"]
        self.group_avg_slope = data_setting["group_avg_slope"]
        self.macro_data = data_setting["macro_data"]
        self.real = data_setting["real"]
        self.total_bw_slope.index = pd.DatetimeIndex(self.total_bw_slope.index)
        self.group_avg_slope.index = pd.DatetimeIndex(self.group_avg_slope.index)
        self.t_dir = data_setting["save_dir"]

    def prior_process(self, n_map, top_n):

        """ prior_process 적용한 방식은 map에서 상위 N을 해당날짜의 구성N집합으로 하여 정답 추세를 이 N집합의 평균기울기와 추세별 평균기울기의 오차 가장 작은 기준으로 설정함 """

        total = pd.DataFrame()
        # n_map.index = pd.DatetimeIndex(n_map.index)
        for date in n_map.index.drop_duplicates():
            t_map = n_map.loc[date].sort_values(by="RANK", ascending=False)[:top_n]
            date_n = ["n_{}".format(i) for i in t_map["N"]]
            ## map에서 각 추세 평균 기울기와 N 집합의 평균 기울기와 가장 오차가 작은 추세를 ANS로 설정한 방식
            try:
                avg_slope = self.total_bw_slope[date_n].loc[date].mean()
                slope_mae = pd.DataFrame(
                    {
                        col: abs(self.group_avg_slope.loc[date, col] - avg_slope)
                        for col in self.group_avg_slope.columns
                    },
                    index=[date],
                )
                total = pd.concat(
                    [
                        total,
                        pd.DataFrame(
                            {
                                "mean_slope": avg_slope,
                                "close_term": slope_mae.loc[date]
                                .sort_values(ascending=True)
                                .index[0],
                            },
                            index=[date],
                        ),
                    ]
                )
                total.to_csv(self.t_dir + "/close_term.csv")
            except:
                pass
        return total

    def post_process(self, pred_term, n_info):
        """ 엔진 결과 후처리 """

        top_n = n_info["top_n"]
        pred_term.index = pd.DatetimeIndex(pred_term.index)

        if len(pred_term.columns) == 1:
            col = pred_term.columns[0]
            total = pd.DataFrame()

            for p_date in pred_term.index:
                t_slope = self.group_avg_slope[pred_term.loc[p_date, col]].loc[p_date]
                mae = pd.DataFrame(
                    {
                        int(i.split("_")[1]): abs(
                            self.total_bw_slope.loc[p_date, i] - t_slope
                        )
                        for i in self.total_bw_slope.columns
                    },
                    index=[p_date],
                )
                total = pd.concat(
                    [
                        total,
                        pd.DataFrame(
                            mae.loc[p_date].sort_values(ascending=True).index[:top_n],
                            columns=["N"],
                            index=[p_date for _ in range(top_n)],
                        ),
                    ]
                )
            total.to_csv(self.t_dir + "/predicted_term_map.csv")

            setting = {"pred_term_map": total, "test_start": "", "save_dir": self.t_dir}
            test, _ = pred_anal(
                self.macro_data, self.real, setting, n_info, predicted_map=True
            ).HIT_MAE(tree_fit=True)
            graph(
                self.t_dir,
                test,
                "../../data/{}gap_{}size".format(n_info["n_gap"], n_info["n_size"]),
            ).monthly_plot()

        else:
            epoch = 20  # map(input, "epoch_check_number")
            for epo in range(epoch):
                tdir = self.t_dir + "ep{}\\".format(epo)
                os.makedirs(tdir, exist_ok=True)
                pt = pred_term[str(epo)]
                total = pd.DataFrame()
                for date in pt.index:
                    t_slope = self.group_avg_slope[pt.loc[date]].loc[date]
                    gv = pd.DataFrame(
                        {
                            i: abs(self.total_bw_slope.loc[date, i] - t_slope)
                            for i in self.total_bw_slope.columns
                        },
                        index=[date],
                    )
                    total = pd.concat(
                        [
                            total,
                            pd.DataFrame(
                                gv.loc[date].sort_values(ascending=True).index[:top_n],
                                columns=["N"],
                                index=[date for _ in range(top_n)],
                            ),
                        ]
                    )

                total.to_csv(tdir + "pred_term_map.csv")
                setting = {"pred_term_map": total, "test_start": "", "save_dir": tdir}
                test, _ = pred_anal(
                    self.macro_data, self.real, setting, n_info, predicted_map=True
                ).HIT_MAE(tree_fit=True)
                graph(
                    tdir, test, "{}gap_{}size".format(n_info["n_gap"], n_info["n_size"])
                ).monthly_plot()
