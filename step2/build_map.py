import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from multiprocessing import Pool
from scipy.stats import entropy as ep
import time
import os


class map_maker:
    def __init__(
        self,
        nat,
        tar,
        name_z,
        z_range,
        top_num,
        term_optN,
        m_stv,
        m_gap,
        m_size,
        n_stv,
        n_gap,
        n_size,
        map_seq_gap,
    ):

        self.nat = nat
        self.tar = tar
        self.name_z = name_z
        self.z_range = z_range
        self.top_num = top_num
        term_optN.index = pd.DatetimeIndex(term_optN.index)
        self.short = term_optN[["S_{}".format(k) for k in range(top_num)]]
        self.mid = term_optN[["M_{}".format(i) for i in range(top_num)]]
        self.long = term_optN[["L_{}".format(i) for i in range(top_num)]]
        self.save_dir = "../data/{}gap_{}size/term_map".format(n_gap, n_size)
        self.m_stv, self.m_gap, self.m_size = m_stv, m_gap, m_size
        self.n_stv, self.n_gap, self.n_size = n_stv, n_gap, n_size
        self.map_seq_gap = map_seq_gap

    def map_build(self, q):
        
        ''' 기간 변수에 대한 Map 생성 '''


def make_ans_target(term_map):

    classify_term = pd.DataFrame()
    for i in term_map.index.drop_duplicates():
        tm = term_map.loc[i]
        orderly = tm.sort_values(by="RANK", ascending=False).iloc[0]
        classify_term = pd.concat(
            [
                classify_term,
                pd.DataFrame(orderly["N"], index=[i], columns=["close_term"]),
            ]
        )

    return classify_term
