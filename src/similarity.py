import math
import pickle
from collections import defaultdict
from typing import Dict

import pandas as pd
from tqdm import tqdm

from .config import SAVE_PATH, I2I_SIM_FILENAME
from .data import get_user_item_time


def itemcf_sim(df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """
    计算文章与文章之间的相似性矩阵（ItemCF）。
    """
    user_item_time_dict = get_user_item_time(df)

    i2i_sim: Dict[int, Dict[int, float]] = {}
    item_cnt = defaultdict(int)

    for _, item_time_list in tqdm(user_item_time_dict.items()):
        for i, _ in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, _ in item_time_list:
                if i == j:
                    continue
                i2i_sim[i].setdefault(j, 0.0)
                i2i_sim[i][j] += 1.0 / math.log(len(item_time_list) + 1)

    i2i_sim_norm: Dict[int, Dict[int, float]] = {}
    for i, related_items in i2i_sim.items():
        i2i_sim_norm[i] = {}
        for j, wij in related_items.items():
            i2i_sim_norm[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 保存
    with open(SAVE_PATH / I2I_SIM_FILENAME, "wb") as f:
        pickle.dump(i2i_sim_norm, f)

    return i2i_sim_norm


