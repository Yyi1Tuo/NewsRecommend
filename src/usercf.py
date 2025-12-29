"""UserCF (用户协同过滤) 召回模块"""
import math
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import SAVE_PATH
from .data import get_user_item_time


def usercf_sim(df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """
    计算用户与用户之间的相似性矩阵（UserCF）
    基于共同点击的文章计算相似度
    """
    # 构建物品到用户的倒排索引
    item_users: Dict[int, set] = defaultdict(set)
    for _, row in df.iterrows():
        item_users[row['click_article_id']].add(row['user_id'])
    
    # 计算用户共现矩阵
    u2u_sim: Dict[int, Dict[int, float]] = {}
    user_cnt = defaultdict(int)
    
    print("计算用户相似度...")
    for item, users in tqdm(item_users.items()):
        users_list = list(users)
        for i, u1 in enumerate(users_list):
            user_cnt[u1] += 1
            u2u_sim.setdefault(u1, {})
            for u2 in users_list[i+1:]:
                user_cnt[u2] += 1
                u2u_sim[u1].setdefault(u2, 0.0)
                u2u_sim.setdefault(u2, {})
                u2u_sim[u2].setdefault(u1, 0.0)
                
                # IUF惩罚（活跃用户贡献度降低）
                u2u_sim[u1][u2] += 1.0 / math.log(len(users) + 1)
                u2u_sim[u2][u1] += 1.0 / math.log(len(users) + 1)
    
    # 归一化
    u2u_sim_norm: Dict[int, Dict[int, float]] = {}
    for u1, related_users in u2u_sim.items():
        u2u_sim_norm[u1] = {}
        for u2, wij in related_users.items():
            u2u_sim_norm[u1][u2] = wij / math.sqrt(user_cnt[u1] * user_cnt[u2])
    
    # 保存
    with open(SAVE_PATH / 'usercf_u2u_sim.pkl', 'wb') as f:
        pickle.dump(u2u_sim_norm, f)
    
    print(f"用户相似度计算完成，共 {len(u2u_sim_norm)} 个用户")
    return u2u_sim_norm


def user_based_recommend(
    user_id: int,
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    u2u_sim: Dict[int, Dict[int, float]],
    sim_user_topk: int,
    recall_item_num: int,
    item_topk_click: List[int],
) -> List[Tuple[int, float]]:
    """
    基于用户协同过滤的召回
    找相似用户喜欢的物品进行推荐
    """
    # 当前用户的历史
    user_hist_items = user_item_time_dict.get(user_id, [])
    user_hist_item_ids = {item_id for item_id, _ in user_hist_items}
    
    # 找相似用户
    if user_id not in u2u_sim:
        # 没有相似用户，返回热门
        result = []
        for idx, item in enumerate(item_topk_click[:recall_item_num]):
            if item not in user_hist_item_ids:
                result.append((item, -idx - 100.0))
        return result
    
    similar_users = sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]
    
    # 聚合相似用户喜欢的物品
    item_rank: Dict[int, float] = {}
    for sim_user, sim_score in similar_users:
        sim_user_items = user_item_time_dict.get(sim_user, [])
        for item_id, _ in sim_user_items:
            if item_id in user_hist_item_ids:
                continue
            item_rank.setdefault(item_id, 0.0)
            item_rank[item_id] += sim_score
    
    # 热门补全
    if len(item_rank) < recall_item_num:
        for idx, item in enumerate(item_topk_click):
            if item in item_rank or item in user_hist_item_ids:
                continue
            item_rank[item] = -idx - 100.0
            if len(item_rank) >= recall_item_num:
                break
    
    # 排序返回
    item_rank_sorted = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
    return item_rank_sorted

