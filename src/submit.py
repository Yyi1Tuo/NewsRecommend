from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import SAVE_PATH


def submit(recall_df: pd.DataFrame, topk: int = 5, model_name: Optional[str] = None) -> Path:
    recall_df = recall_df.sort_values(by=["user_id", "pred_score"])  # 排序后分组排名
    recall_df["rank"] = recall_df.groupby(["user_id"]) ["pred_score"].rank(ascending=False, method="first")

    # 每个用户至少 topk 个
    tmp = recall_df.groupby("user_id").apply(lambda x: x["rank"].max())
    assert tmp.min() >= topk

    recall_df = recall_df.copy()
    del recall_df["pred_score"]
    submit_df = (
        recall_df[recall_df["rank"] <= topk]
        .set_index(["user_id", "rank"]).unstack(-1).reset_index()
    )

    submit_df.columns = [int(col) if isinstance(col, int) else col for col in submit_df.columns.droplevel(0)]
    
    # 动态构建列名
    rename_dict = {"": "user_id"}
    for i in range(1, topk + 1):
        rename_dict[i] = f"article_{i}"
    submit_df = submit_df.rename(columns=rename_dict)

    if model_name is None:
        model_name = "itemcf"
    save_name = SAVE_PATH / "submits" / f"{model_name}_" f"{datetime.today().strftime('%m-%d-%H-%M-%S')}.csv"
    submit_df.to_csv(save_name, index=False, header=True)
    return save_name


