import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class pred_anal:

    def __init__(self, macro_data, real_data, setting, n_info, predicted_map=False):

        self.data = macro_data
        self.data.index = pd.DatetimeIndex(self.data.index)
        self.real = real_data
        self.real.index = pd.DatetimeIndex(self.real.index)
        self.save_dir = setting["save_dir"]

        if predicted_map == False:
            self.pred = setting["pred_price"]
            self.pred.index = pd.DatetimeIndex(self.pred.index)
            term_map = setting["term_map"]
            self.term_map = term_map[term_map["RANK"] > setting["rank_bottom"]]
            self.term_map.index = pd.DatetimeIndex(self.term_map.index)
            self.term_info = setting["each_term_top_N_match"]
            if self.term_info != "":
                self.term_info.index = pd.DatetimeIndex(self.term_info.index)
            if setting["test_start"] == "":
                self.test_date = pd.date_range(
                    self.pred.index[0], self.pred.index[-1], freq="BM"
                )[
                    :-1
                ]  # pd.date_range(start='2015-12-29', end='2018-06-29', freq='BM')
            else:
                self.test_date = pd.date_range(
                    setting["test_start"], self.pred.index[-1], freq="BM"
                )[:-1]

        else:
            self.term_map = setting["pred_term_map"]
            self.term_map.index = pd.DatetimeIndex(self.term_map.index)
            if setting["test_start"] == "":
                self.test_date = pd.date_range(
                    self.term_map.index[0], self.term_map.index[-1], freq="BM"
                )[
                    :-1
                ]  # pd.date_range(start='2015-12-29', end='2018-06-29', freq='BM')
            else:
                self.test_date = pd.date_range(
                    setting["test_start"], self.term_map.index[-1], freq="BM"
                )[:-1]

        self.top_n = n_info["top_n"]
        self.n_stv = n_info["n_stv"]
        self.n_gap = n_info["n_gap"]
        self.n_size = n_info["n_size"]
        self.predicted_map = predicted_map

    def tree_predict(self, p_date):

        ''' 예측 검정시 Tree Fitting for index price '''

    def HIT_MAE(self, tree_fit):

        total = pd.DataFrame()
        n_info = pd.DataFrame()
        for rebal_date in self.test_date:

            if tree_fit == True:
                n_set, pred_table = self.tree_predict(rebal_date)
                total = pd.concat([total, pred_table])
                n_info = pd.concat([n_info, n_set])
            else:
                start = pd.date_range(rebal_date, periods=2, freq="B")[-1]
                end = pd.date_range(start, periods=1, freq="BM")[0]
                pred_date = pd.date_range(start, end, freq="B")
                tt = [str(i) for i in self.term_map.loc[rebal_date]["N"]]
                total = pd.concat(
                    [
                        total,
                        pd.DataFrame(
                            self.pred[tt]
                            .loc[start][: len(pred_date)]
                            .mean(axis=1)
                            .values,
                            index=pred_date,
                            columns=["pred"],
                        ),
                    ]
                )
        total = pd.concat([total, self.real], axis=1).dropna()

        if tree_fit == True:
            total.to_csv(self.save_dir + "fitted_optimal_pred.csv")
            n_info.columns = ["N"]
            n_info.to_csv(self.save_dir + "fitted_optimal_N.csv")
        else:
            total.to_csv(self.save_dir + "optimal_N.csv")

        bms = pd.date_range(start=self.test_date[0], end=self.test_date[-1], freq="BMS")
        bm = pd.date_range(start=self.test_date[0], end=self.test_date[-1], freq="BM")
        if bms[0] > bm[0]:
            bm = bm.drop(bm[0])
        if bms[-1] > bm[-1]:
            bms = bms.drop(bms[-1])
        info = pd.DataFrame()
        for st, end in zip(bms, bm):
            t_real, t_pred = total["spx_index"].loc[st:end], total["pred"].loc[st:end]
            mae = abs(t_real - t_pred).mean()
            m_mae = abs(t_real[-2:] - t_pred[-2:]).mean()
            ret = t_real.pct_change() * t_pred.pct_change()
            ret[ret >= 0] = 1
            ret[ret < 0] = 0
            hit = ret.dropna().mean()
            real_change = t_real.iloc[-1] - t_real.iloc[0]
            pred_change = t_pred.iloc[-1] - t_pred.iloc[0]
            if real_change * pred_change >= 0:
                mh = 1
            else:
                mh = 0
            info = pd.concat(
                [
                    info,
                    pd.DataFrame(
                        {"MAE": mae, "M_MAE": m_mae, "HIT": hit, "M_HIT": mh},
                        index=[st],
                    ),
                ]
            )

        print(info.mean())
        if tree_fit == False:
            info.mean().to_csv(self.save_dir + "HIT_and_MAE.csv")
        else:
            info.mean().to_csv(self.save_dir + "fitted_HIT_and_MAE.csv")
        return total, info.mean()
