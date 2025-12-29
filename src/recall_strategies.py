"""多种召回策略实现"""
import math
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd


def hot_recall(
    user_hist_items: Set[int],
    item_topk_click: List[int],
    recall_num: int = 10
) -> List[Tuple[int, float]]:
    """
    热门召回：直接推荐最热门的文章
    """
    result = []
    for idx, item in enumerate(item_topk_click):
        if item not in user_hist_items:
            result.append((item, 1000.0 - idx))  # 热度分数
            if len(result) >= recall_num:
                break
    return result


def time_decay_recall(
    user_id: int,
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    all_click_df: pd.DataFrame,
    recall_num: int = 10,
    decay_days: int = 7
) -> List[Tuple[int, float]]:
    """
    时间衰减召回：最近热门的文章权重更高
    """
    # 获取用户历史
    user_hist_items = user_item_time_dict.get(user_id, [])
    user_hist_item_ids = {item_id for item_id, _ in user_hist_items}
    
    # 计算每个文章的时间衰减热度
    max_time = all_click_df['click_timestamp'].max()
    item_scores: Dict[int, float] = {}
    
    for _, row in all_click_df.iterrows():
        item_id = row['click_article_id']
        timestamp = row['click_timestamp']
        
        if item_id in user_hist_item_ids:
            continue
        
        # 指数衰减
        time_diff_days = (max_time - timestamp) / 86400  # 转为天数
        decay_weight = math.exp(-time_diff_days / decay_days)
        
        item_scores.setdefault(item_id, 0.0)
        item_scores[item_id] += decay_weight
    
    # 排序返回
    item_rank_sorted = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:recall_num]
    return item_rank_sorted


def diversity_recall(
    candidate_items: List[Tuple[int, float]],
    item_features: pd.DataFrame,
    diversity_weight: float = 0.3,
    topk: int = 10
) -> List[Tuple[int, float]]:
    """
    多样性召回：在保证相关性的同时增加多样性
    使用 MMR (Maximal Marginal Relevance) 算法
    """
    if len(candidate_items) <= topk:
        return candidate_items
    
    # 归一化分数
    scores = np.array([score for _, score in candidate_items])
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    # 构建物品特征向量（简化版：使用点击统计特征）
    item_ids = [item_id for item_id, _ in candidate_items]
    features_dict = {}
    for item_id in item_ids:
        if item_id in item_features['click_article_id'].values:
            feat = item_features[item_features['click_article_id'] == item_id].iloc[0]
            features_dict[item_id] = np.array([
                feat.get('click_count', 0),
                feat.get('unique_users', 0),
                feat.get('time_popularity', 0)
            ])
        else:
            features_dict[item_id] = np.zeros(3)
    
    # MMR 选择
    selected = []
    remaining = list(range(len(candidate_items)))
    
    # 先选相关性最高的
    best_idx = np.argmax(scores)
    selected.append(remaining.pop(best_idx))
    
    # 迭代选择
    while len(selected) < topk and remaining:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining:
            item_id = candidate_items[idx][0]
            relevance = scores[idx]
            
            # 计算与已选物品的最大相似度
            max_sim = 0.0
            for sel_idx in selected:
                sel_item_id = candidate_items[sel_idx][0]
                # 余弦相似度
                feat1 = features_dict[item_id]
                feat2 = features_dict[sel_item_id]
                norm1 = np.linalg.norm(feat1)
                norm2 = np.linalg.norm(feat2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(feat1, feat2) / (norm1 * norm2)
                    max_sim = max(max_sim, sim)
            
            # MMR 分数
            mmr_score = diversity_weight * relevance - (1 - diversity_weight) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(remaining.pop(remaining.index(best_idx)))
    
    # 返回选中的物品
    result = [candidate_items[idx] for idx in selected]
    return result


def cold_start_recall(
    user_id: int,
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    item_topk_click: List[int],
    all_click_df: pd.DataFrame,
    recall_num: int = 10
) -> List[Tuple[int, float]]:
    """
    冷启动召回：针对新用户的召回策略
    结合热门和时间衰减
    """
    user_hist_items = user_item_time_dict.get(user_id, [])
    
    # 如果用户行为少于3次，认为是新用户
    if len(user_hist_items) >= 3:
        return []
    
    user_hist_item_ids = {item_id for item_id, _ in user_hist_items}
    
    # 50%热门 + 50%时间衰减
    hot_items = hot_recall(user_hist_item_ids, item_topk_click, recall_num // 2)
    time_items = time_decay_recall(user_id, user_item_time_dict, all_click_df, recall_num // 2)
    
    # 合并去重
    result_dict = {}
    for item, score in hot_items:
        result_dict[item] = score * 0.6  # 热门权重
    
    for item, score in time_items:
        if item in result_dict:
            result_dict[item] += score * 0.4  # 时间权重
        else:
            result_dict[item] = score * 0.4
    
    result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[:recall_num]
    return result

