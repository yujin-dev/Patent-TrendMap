import pandas as pd
import numpy as np
import os

from multiprocessing import Pool
from collections import OrderedDict
from step1.cal_slope import big3_trend
from step2.build_map import map_maker, make_ans_target
from step3.model_run import model_run
from step3.processing.group_process import group_processing


if __name__ == "__main__":

    n_set = list(range(1, 25))
    n_set.extend(list(range(26, 50, 2)))
    n_info1 = {"n_stv": 25, "n_gap": 10, "n_size": len(n_set), "n_set": n_set}
    n_set = list(range(1, 14))
    n_set.extend(list(range(15, 24, 2)))
    n_info2 = {"n_stv": 25, "n_gap": 10, "n_size": len(n_set), "n_set": n_set}
    n_set = list(range(1, 14))
    n_set.extend(list(range(15, 24, 2)))
    n_info3 = {"n_stv": 25, "n_gap": 22, "n_size": len(n_set), "n_set": n_set}
    n_set = list(range(1, 19))
    n_set.extend(list(range(19, 36, 2)))
    n_info4 = {"n_stv": 25, "n_gap": 10, "n_size": len(n_set), "n_set": n_set}

    info_num = input("n_info = ?")
    if info_num == "1":
        n_info = n_info1
    elif info_num == "2":
        n_info = n_info2
    elif info_num == "3":
        n_info = n_info3
    elif info_num == "4":
        n_info = n_info4
    print("N spec : ", n_info)
    dir = "data/{}gap_{}size".format(n_info["n_gap"], n_info["n_size"])

    spx_index = pd.read_csv("data/spx_index_daily.csv", index_col=0)
    get_trend = big3_trend(spx_index, n_info)
    step1 = input("slope 계산 : yes or no ?")
    if step1 == "yes":
        print("################# Step1 - slope 계산 #################")
        ## Multiprocessing
        pool_set = [x for x in range(0, n_info["n_size"], int(n_info["n_size"] / 3))]
        pool = Pool(processes=len(pool_set))
        pool.map(get_trend.cal_slope, pool_set)

    param = {
        "nat": "US",
        "tar": "SPX",  # OAS, TTR
        "name_z": "ESL",  # ESL, KLD
        "z_range": "all",  # all, lst
        "top_num": 4,
        "map_seq_gap": 500,
        "m_stv": 250,
        "m_gap": 1,
        "m_size": 1,
        "n_stv": n_info["n_stv"],
        "n_gap": n_info["n_gap"],
        "n_size": n_info["n_size"],
    }

    if "{}_total_bw_mean_{}clust.csv".format(
        "gm_slope", n_info["n_size"]
    ) not in os.listdir(dir):
        get_trend.total_slope(how="gm_slope", toward="bw")

    param["term_optN"] = pd.read_csv(dir + "/each_term_match_top_N.csv", index_col=0)
    total_bw_slope = pd.read_csv(
        dir + "/{}_total_bw_mean_{}clust.csv".format("gm_slope", param["n_size"]),
        index_col=0,
    )
    group_avg_slope = pd.read_csv(
        dir + "/{}_group_avg_slope_{}clust.csv".format("gm_slope", param["n_size"]),
        index_col=0,
    )

    if step1 != "yes":
        step2 = input("Map 생성 : yes or no ?")
    else:
        step2 = "yes"
    if step2 == "yes":
        print("################# Step2 - Map 생성 #################")
        date = [i for i in pd.read_csv("data/TTR_SPX_all.csv", index_col=0).index]
        start_num = date.index(
            pd.date_range(end="2007-01-31", periods=23, freq="B")[0].strftime(
                "%Y-%m-%d"
            )
        )
        end_num = date.index(
            pd.date_range(end="2018-06-29", periods=23, freq="B")[0].strftime(
                "%Y-%m-%d"
            )
        )
        map_dates = []
        for x in range(start_num, end_num, param["map_seq_gap"]):
            if x + param["map_seq_gap"] > end_num:
                map_dates.append([x, end_num])
            else:
                map_dates.append([x, x + param["map_seq_gap"]])
        cpu_num = len(map_dates)
        pool = Pool(processes=cpu_num)
        pool.map(map_maker(**param).map_build, map_dates)
        save_dir = map_maker(**param).save_dir
        for (dirpath, dirnames, filenames) in os.walk(save_dir):
            break

        ans = [i for i in filenames if "MAP" in i]
        preds = [i for i in filenames if "PRED" in i]
        ans = list(OrderedDict.fromkeys(ans))
        preds = list(OrderedDict.fromkeys(preds))

        for num, did in enumerate([ans, preds]):
            m_data = pd.DataFrame()
            for i in did:
                data = pd.read_csv(save_dir + "\\" + i, index_col=0)
                m_data = pd.concat([m_data, data], axis=0)
            if num == 0:
                m_data.to_csv(dir + "/TERM_MAP.csv")
                make_ans_target(m_data).to_csv(dir + "/close_term.csv")
            else:
                m_data.to_csv(dir + "/PRED.csv")

    macro_input = pd.read_csv("data/TTR_SPX_all.csv", index_col=0)
    term_target = pd.read_csv(dir + "/close_term.csv", index_col=0)
    setting = {
        "RBIN": "split_online",  # '', 'SLP', 'RBIN', 'RANK', # split, split_online, online
        "Seq": 90,
        "Fit": 1,
        "shuffle": True,
        "E_node": [4000],
        "D_node": [4000],
        "dropout": 0.8,
        "lr_rate": 0.001,
        "epoch": 100,
        "Tbatch": 20,
        "Vbatch": 0,
        "val": 0,
        "te": 0.22,
    }
    key = list(setting.keys())[0]
    if key not in ["", "SLP"]:
        setting["Fit"] = 23

    if step1 == "yes" or step2 == "yes":
        step3 = "yes"  # input('모델 학습 : yes or no ?')
    else:
        step3 = input("모델 학습 : yes or no ?")
    if step3 == "yes":
        print("################# Step3 - Learning #################")

        print("Input Data : {}".format(key))
        if key == "":
            input = macro_input
        elif key == "SLP":
            input = total_bw_slope
        else:  # RBIN, RANK
            data = pd.read_csv(
                dir
                + "/ESL_RANK_and_RBIN_newmap_m{}_n{}.csv".format(1, n_info["n_size"]),
                index_col=0,
            )
            input = np.reshape(data[key].values, [-1, n_info["n_size"]])
            input = pd.DataFrame(
                input,
                index=data.index.drop_duplicates(),
                columns=range(n_info["n_size"]),
            )
        print("Engine spec : ", setting)
        model_run(setting, input, term_target, dir).run()  # macro_input

    t_dir = model_run(setting, macro_input, term_target, dir).save_dir + "\\"
    if step1 == "yes" or step2 == "yes" or step3 == "yes":
        step4 = "yes"
    else:
        step4 = input("최종 예측 : yes or no ?")
    if step4 == "yes":
        print("################# Step4 - Engine Map Processing #################")
        pred_term = pd.read_csv(t_dir + "PRED_TERM.csv", index_col=0)
        data_setting = {
            "total_bw_slope": total_bw_slope,
            "group_avg_slope": group_avg_slope,
            "macro_data": macro_input,
            "real": spx_index,
            "save_dir": t_dir,
        }
        group_processing(data_setting).post_process(
            pred_term,
            {
                "n_stv": n_info["n_stv"],
                "n_gap": n_info["n_gap"],
                "n_size": n_info["n_size"],
                "top_n": 4,
            },
        )
