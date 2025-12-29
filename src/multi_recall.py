"""多路召回融合策略"""
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def weighted_fusion(
    recall_results: Dict[str, List[Tuple[int, float]]],
    weights: Dict[str, float] = None
) -> List[Tuple[int, float]]:
    """
    加权融合多路召回结果
    
    Args:
        recall_results: {召回策略名: [(item_id, score), ...]}
        weights: {召回策略名: 权重}
    
    Returns:
        融合后的推荐列表 [(item_id, final_score), ...]
    """
    if weights is None:
        # 默认权重
        weights = {
            'itemcf': 0.3,
            'usercf': 0.25,
            'two_tower': 0.25,
            'hot': 0.1,
            'time_decay': 0.05,
            'cold_start': 0.05
        }
    
    # 归一化每路召回的分数
    normalized_results = {}
    for strategy, items in recall_results.items():
        if not items:
            continue
        
        scores = np.array([score for _, score in items])
        if len(scores) > 0 and scores.max() > scores.min():
            # 归一化到 [0, 1]
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = np.ones(len(scores))
        
        normalized_results[strategy] = [
            (item_id, float(norm_score))
            for (item_id, _), norm_score in zip(items, normalized_scores)
        ]
    
    # 加权融合
    item_scores: Dict[int, float] = defaultdict(float)
    item_count: Dict[int, int] = defaultdict(int)
    
    for strategy, items in normalized_results.items():
        weight = weights.get(strategy, 0.1)
        for item_id, score in items:
            item_scores[item_id] += score * weight
            item_count[item_id] += 1
    
    # 按分数排序
    final_results = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return final_results


def rank_fusion(
    recall_results: Dict[str, List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    基于排名的融合 (Reciprocal Rank Fusion)
    不依赖原始分数，只看排名
    
    RRF score = sum(1 / (k + rank_i))
    """
    item_scores: Dict[int, float] = defaultdict(float)
    
    for strategy, items in recall_results.items():
        if not items:
            continue
        
        for rank, (item_id, _) in enumerate(items, start=1):
            # RRF 公式
            item_scores[item_id] += 1.0 / (k + rank)
    
    # 排序
    final_results = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return final_results


def cascade_fusion(
    recall_results: Dict[str, List[Tuple[int, float]]],
    strategy_order: List[str] = None,
    target_num: int = 100
) -> List[Tuple[int, float]]:
    """
    级联融合：按优先级依次选择
    先从高优先级策略选，不足再从低优先级补充
    """
    if strategy_order is None:
        strategy_order = ['two_tower', 'itemcf', 'usercf', 'time_decay', 'hot', 'cold_start']
    
    selected_items = {}
    
    for strategy in strategy_order:
        if strategy not in recall_results:
            continue
        
        items = recall_results[strategy]
        for item_id, score in items:
            if item_id not in selected_items:
                selected_items[item_id] = score
                if len(selected_items) >= target_num:
                    break
        
        if len(selected_items) >= target_num:
            break
    
    # 按分数排序
    final_results = sorted(selected_items.items(), key=lambda x: x[1], reverse=True)
    return final_results


def hybrid_fusion(
    recall_results: Dict[str, List[Tuple[int, float]]],
    fusion_mode: str = 'weighted',
    **kwargs
) -> List[Tuple[int, float]]:
    """
    混合融合：根据模式选择不同的融合策略
    
    Args:
        recall_results: 多路召回结果
        fusion_mode: 'weighted'(加权), 'rank'(排名), 'cascade'(级联)
        **kwargs: 传递给具体融合函数的参数
    
    Returns:
        融合后的推荐列表
    """
    if fusion_mode == 'weighted':
        return weighted_fusion(recall_results, weights=kwargs.get('weights'))
    elif fusion_mode == 'rank':
        return rank_fusion(recall_results, k=kwargs.get('k', 60))
    elif fusion_mode == 'cascade':
        return cascade_fusion(
            recall_results,
            strategy_order=kwargs.get('strategy_order'),
            target_num=kwargs.get('target_num', 100)
        )
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")


def filter_and_rerank(
    fused_results: List[Tuple[int, float]],
    user_hist_items: set,
    item_features: dict = None,
    topk: int = 100
) -> List[Tuple[int, float]]:
    """
    过滤和重排序
    - 过滤用户已看过的
    - 可选：基于其他规则重排（如多样性、新鲜度）
    """
    # 过滤历史
    filtered = [(item, score) for item, score in fused_results if item not in user_hist_items]
    
    # TODO: 可以加入更多重排逻辑，如：
    # - 多样性惩罚
    # - 新鲜度加权
    # - 业务规则过滤
    
    return filtered[:topk]

