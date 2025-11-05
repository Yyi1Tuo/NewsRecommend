import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    DATA_PATH,
    SAVE_PATH,
    I2I_SIM_FILENAME,
    SIM_ITEM_TOPK,
    RECALL_ITEM_NUM,
    ITEM_TOPK_K,
    MODEL_NAME,
)
from . import data as data_mod
from . import similarity
from . import recall
from . import submit as submit_mod


def _load_or_build_i2i(all_click_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    sim_path = SAVE_PATH / I2I_SIM_FILENAME
    if sim_path.exists():
        with open(sim_path, "rb") as f:
            return pickle.load(f)
    return similarity.itemcf_sim(all_click_df)


def run(topk_submit: int = 5) -> Path:
    # 读取全量（训练+测试）点击数据
    all_click_df = data_mod.get_all_click_df(offline=False)

    # 相似度矩阵
    i2i_sim = _load_or_build_i2i(all_click_df)

    # 用户历史与热门文章
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K)

    # 召回
    user_recall_items_dict: Dict[int, List[Tuple[int, float]]] = defaultdict[int, List[Tuple[int, float]]](list) # 类型注解：字典[用户ID, 推荐文章列表]
    for user in tqdm(all_click_df["user_id"].unique()):
        recs = recall.item_based_recommend(
            user_id=user,
            user_item_time_dict=user_item_time_dict,
            i2i_sim=i2i_sim,
            sim_item_topk=SIM_ITEM_TOPK,
            recall_item_num=RECALL_ITEM_NUM,
            item_topk_click=item_topk_click,
        )
        user_recall_items_dict[user] = recs

    # to DataFrame
    user_item_score_list: List[Tuple[int, int, float]] = []
    for user, items in user_recall_items_dict.items():
        for item, score in items:
            user_item_score_list.append([user, item, score])
    recall_df = pd.DataFrame(user_item_score_list, columns=["user_id", "click_article_id", "pred_score"])

    # 测试集过滤
    tst_click = pd.read_csv(str(DATA_PATH / "testA_click_log.csv"))
    tst_users = tst_click["user_id"].unique()
    tst_recall = recall_df[recall_df["user_id"].isin(tst_users)]

    # 生成提交
    submit_path = submit_mod.submit(tst_recall, topk=topk_submit, model_name=MODEL_NAME)
    return submit_path


