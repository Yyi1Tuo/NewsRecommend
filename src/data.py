import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
from .config import DATA_PATH


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    starttime = time.time()
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        "-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )
    return df


def get_all_click_sample(data_path: str, sample_nums: int = 10000) -> pd.DataFrame:
    all_click = pd.read_csv(str(DATA_PATH / "train_click_log.csv"))
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click["user_id"].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"]) 
    return all_click


def get_all_click_df(data_path: str = None, offline: bool = True) -> pd.DataFrame:
    base = DATA_PATH if data_path is None else Path(data_path)
    if offline:
        all_click = pd.read_csv(str(base / "train_click_log.csv"))
    else:
        trn_click = pd.read_csv(str(base / "train_click_log.csv"))
        tst_click = pd.read_csv(str(base / "testA_click_log.csv"))
        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"]) 
    return all_click


def get_user_item_time(click_df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    click_df = click_df.sort_values("click_timestamp")

    def make_item_time_pair(df: pd.DataFrame) -> List[Tuple[int, int]]:
        return list(zip(df["click_article_id"], df["click_timestamp"]))

    user_item_time_df = (
        click_df.groupby("user_id")[
            ["click_article_id", "click_timestamp"]
        ].apply(lambda x: make_item_time_pair(x))
        .reset_index()
        .rename(columns={0: "item_time_list"})
    )
    user_item_time_dict = dict(
        zip(user_item_time_df["user_id"], user_item_time_df["item_time_list"]) 
    )
    return user_item_time_dict


def get_item_topk_click(click_df: pd.DataFrame, k: int):
    topk_click = click_df["click_article_id"].value_counts().index[:k]
    return topk_click


