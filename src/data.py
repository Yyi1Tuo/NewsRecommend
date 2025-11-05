import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
from .config import DATA_PATH


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """下采样数值列以减少内存占用。

    思路：
    - 根据每列最小/最大值选择最小可容纳的整数/浮点精度类型
    - 忽略存在缺失但无法安全转换的列（直接跳过）
    - 返回转换后的 DataFrame（原地修改）
    """
    starttime = time.time()
    # 仅对这些数值类型尝试压缩
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            # 估计边界以选择更低位宽的类型
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
    """从训练集按用户随机采样一部分数据用于开发调试。"""
    all_click = pd.read_csv(str(DATA_PATH / "train_click_log.csv"))
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click["user_id"].isin(sample_user_ids)]

    # 去重键：同一用户、同一文章、同一时间戳视为一条点击
    all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"]) 
    return all_click


def get_all_click_df(data_path: str = None, offline: bool = True) -> pd.DataFrame:
    """读取点击日志。

    参数：
    - data_path：可选自定义数据目录；None 时使用配置中的 `DATA_PATH`
    - offline=True：仅使用训练集；False：训练集与测试集合并（用于线上提交）
    """
    # 兼容外部自定义路径；默认使用配置目录
    base = DATA_PATH if data_path is None else Path(data_path)
    if offline:
        all_click = pd.read_csv(str(base / "train_click_log.csv"))
    else:
        trn_click = pd.read_csv(str(base / "train_click_log.csv"))
        tst_click = pd.read_csv(str(base / "testA_click_log.csv"))
        # 线上模式：训练+测试拼接（pandas>=2.0 移除了 DataFrame.append，使用 concat）
        all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    # 去重，防止重复点击记录影响统计
    all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"]) 
    return all_click


def get_user_item_time(click_df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    """构建用户历史点击序列：{user_id: [(item_id, timestamp), ...]}"""
    click_df = click_df.sort_values("click_timestamp")

    def make_item_time_pair(df: pd.DataFrame) -> List[Tuple[int, int]]:
        # 将分组内两列压缩为 (item_id, ts) 元组列表
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
    """获取全局点击次数最高的前 k 篇文章（用于热门补全）。"""
    topk_click = click_df["click_article_id"].value_counts().index[:k]
    return topk_click


